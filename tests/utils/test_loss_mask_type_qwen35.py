import pytest

from slime.utils.mask_utils import MultiTurnLossMaskGenerator

NUM_GPUS = 0


class FakeQwen35Tokenizer:
    """A tiny char-level tokenizer that models the Qwen3.5 assistant formatting rule.

    The critical behavior we need for these tests is:
    1. `add_generation_prompt=True` appends `<|im_start|>assistant\n<think>\n`
    2. Only assistant turns after the last non-tool user query are wrapped in
       `<think>...</think>`
    """

    assistant_generation_prompt = "<|im_start|>assistant\n<think>\n"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoded = {"input_ids": [ord(ch) for ch in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        tools=None,
        add_generation_prompt=False,
        return_dict=False,
        add_special_tokens=False,
        **kwargs,
    ):
        rendered = self.render(messages, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return [ord(ch) for ch in rendered]
        return rendered

    def render(self, messages, add_generation_prompt=False):
        rendered, _ = self.render_with_expected_mask(messages, add_generation_prompt=add_generation_prompt)
        return rendered

    def render_with_expected_mask(self, messages, add_generation_prompt=False):
        if not messages:
            raise ValueError("No messages provided.")

        pieces = []
        mask = []
        last_query_index = self._find_last_query_index(messages)
        has_assistant = any(message["role"] == "assistant" for message in messages)
        if has_assistant and last_query_index is None:
            raise ValueError("No user query found in messages.")

        for index, message in enumerate(messages):
            role = message["role"]
            if role == "system":
                if index != 0:
                    raise ValueError("System message must be at the beginning.")
                piece = f"<|im_start|>system\n{message['content']}<|im_end|>\n"
                pieces.append(piece)
                mask.extend([0] * len(piece))
                continue

            if role == "user":
                piece = f"<|im_start|>user\n{message['content']}<|im_end|>\n"
                pieces.append(piece)
                mask.extend([0] * len(piece))
                continue

            if role == "tool":
                piece = ""
                if index > 0 and messages[index - 1]["role"] != "tool":
                    piece += "<|im_start|>user"
                piece += f"\n<tool_response>\n{message['content']}\n</tool_response>"
                if index == len(messages) - 1 or messages[index + 1]["role"] != "tool":
                    piece += "<|im_end|>\n"
                pieces.append(piece)
                mask.extend([0] * len(piece))
                continue

            if role != "assistant":
                raise NotImplementedError(f"Unsupported role in test tokenizer: {role}")

            reasoning, answer = self._split_assistant_content(message["content"])
            if index > last_query_index:
                prefix = "<|im_start|>assistant\n<think>\n"
                target = f"{reasoning}\n</think>\n\n{answer}"
            else:
                prefix = "<|im_start|>assistant\n"
                target = answer

            target += self._render_tool_calls(answer, message.get("tool_calls"))
            target += "<|im_end|>\n"

            pieces.append(prefix + target)
            mask.extend([0] * len(prefix))
            mask.extend([1] * len(target))

        if add_generation_prompt:
            pieces.append(self.assistant_generation_prompt)
            mask.extend([0] * len(self.assistant_generation_prompt))

        return "".join(pieces), mask

    @staticmethod
    def _split_assistant_content(content):
        if "</think>" not in content:
            return "", content
        reasoning = content.split("</think>")[0].split("<think>")[-1].strip("\n")
        answer = content.split("</think>")[-1].lstrip("\n")
        return reasoning, answer

    @staticmethod
    def _render_tool_calls(answer, tool_calls):
        if not tool_calls:
            return ""

        pieces = []
        for index, tool_call in enumerate(tool_calls):
            function_call = tool_call.get("function", tool_call)
            function_name = function_call["name"]
            if index == 0:
                if answer.strip():
                    pieces.append(f"\n\n<tool_call>\n<function={function_name}>\n")
                else:
                    pieces.append(f"<tool_call>\n<function={function_name}>\n")
            else:
                pieces.append(f"\n<tool_call>\n<function={function_name}>\n")

            for argument_name, argument_value in function_call.get("arguments", {}).items():
                pieces.append(f"<parameter={argument_name}>\n")
                pieces.append(str(argument_value))
                pieces.append("\n</parameter>\n")

            pieces.append("</function>\n</tool_call>")

        return "".join(pieces)

    @staticmethod
    def _find_last_query_index(messages):
        last_query_index = None
        for index, message in enumerate(messages):
            if message["role"] == "user":
                last_query_index = index
        return last_query_index


def test_qwen3_and_qwen3_5_on_single_turn_qwen35_data():
    tokenizer = FakeQwen35Tokenizer()
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": "<think>REASONING</think>\nANSWER"},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    token_ids, loss_mask = generator.get_loss_mask(messages)
    assert token_ids == expected_token_ids
    assert loss_mask == expected_mask

    # qwen3_5 trains what rollout samples: from after the generation prompt through
    # <|im_end|>, without the newline the template writes after it.
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")
    token_ids, loss_mask = generator.get_loss_mask(messages)
    assert token_ids == expected_token_ids
    assert generator.get_text_from_loss_mask(token_ids, loss_mask) == ["REASONING\n</think>\n\nANSWER<|im_end|>"]


def test_qwen3_and_qwen3_5_diverge_on_multi_turn_qwen35_data():
    tokenizer = FakeQwen35Tokenizer()
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER_1"},
        {"role": "assistant", "content": "ANSWER_1"},
        {"role": "user", "content": "USER_2"},
        {"role": "assistant", "content": "<think>REASONING_2</think>\nANSWER_2"},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    qwen3_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    qwen35_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")

    qwen3_token_ids, qwen3_loss_mask = qwen3_generator.get_loss_mask(messages)

    # `qwen3` is incompatible with the real Qwen3.5 full-text rendering:
    # it rebuilds each assistant turn in isolation and fabricates a `<think>` block
    # for the earlier assistant answer.
    assert qwen3_token_ids != expected_token_ids
    assert qwen3_loss_mask != expected_mask

    assert qwen3_generator.get_text_from_loss_mask(qwen3_token_ids, qwen3_loss_mask) == [
        "\n</think>\n\nANSWER_1<|im_end|>\n",
        "REASONING_2\n</think>\n\nANSWER_2<|im_end|>\n",
    ]
    # The template renders ANSWER_1 without a thinking block once USER_2 exists. Rollout
    # samples every turn after `<|im_start|>assistant\n<think>\n`, so that rendering was never
    # sampled; qwen3_5 refuses it instead of training it.
    with pytest.raises(ValueError, match="Assistant message 2"):
        qwen35_generator.get_loss_mask(messages)


def test_qwen3_5_matches_expected_mask_for_tool_call_flow():
    tokenizer = FakeQwen35Tokenizer()
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
        {
            "role": "assistant",
            "content": "TOOL_CALL",
            "tool_calls": [{"function": {"name": "terminal", "arguments": {"command": "ls"}}}],
        },
        {"role": "tool", "content": "README.md"},
        {"role": "assistant", "content": "<think>REASONING</think>\nFINAL"},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")
    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert token_ids == expected_token_ids
    assert generator.get_text_from_loss_mask(token_ids, loss_mask) == [
        "\n</think>\n\nTOOL_CALL\n\n<tool_call>\n<function=terminal>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call><|im_end|>",
        "REASONING\n</think>\n\nFINAL<|im_end|>",
    ]


def test_qwen3_matches_full_template_for_consecutive_tool_responses():
    tokenizer = FakeQwen35Tokenizer()
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
        {
            "role": "assistant",
            "content": "CALL",
            "tool_calls": [
                {"function": {"name": "terminal", "arguments": {"command": "cat a.py"}}},
                {"function": {"name": "terminal", "arguments": {"command": "cat b.py"}}},
            ],
        },
        {"role": "tool", "content": "CONTENTS_A"},
        {"role": "tool", "content": "CONTENTS_B"},
    ]

    expected_text, expected_loss_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")

    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert token_ids == expected_token_ids
    assert loss_mask == expected_loss_mask



class Qwen35TemplateTokenizer:
    """A tiny char-level tokenizer that models the Qwen3.5 chat template.

    1. `add_generation_prompt=True` appends `<|im_start|>assistant\n<think>\n`, or, with
       `enable_thinking=False`, `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
    2. An assistant turn rendered with a thinking block writes
       `<think>\n{reasoning_content}\n</think>\n\n` before its content (an empty block when
       there is no reasoning). The model's own template does so only after the last user
       query; `preserve_thinking=True` (a template variant) does so for every turn.
    3. Every turn ends with `<|im_end|>\n`, and a conversation with an assistant turn but no
       user query is refused, as the real template does.
    """

    def __init__(self, preserve_thinking=False):
        self.preserve_thinking = preserve_thinking

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoded = {"input_ids": [ord(ch) for ch in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        tools=None,
        add_generation_prompt=False,
        return_dict=False,
        add_special_tokens=False,
        enable_thinking=True,
        **kwargs,
    ):
        rendered = self.render(messages, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
        if tokenize:
            return [ord(ch) for ch in rendered]
        return rendered

    def render(self, messages, add_generation_prompt=False, enable_thinking=True):
        users = [index for index, message in enumerate(messages) if message["role"] == "user"]
        if any(message["role"] == "assistant" for message in messages) and not users:
            raise ValueError("No user query found in messages.")
        last_query_index = users[-1] if users else -1
        pieces = []
        for index, message in enumerate(messages):
            role = message["role"]
            if role in ("system", "user"):
                pieces.append(f"<|im_start|>{role}\n{message['content']}<|im_end|>\n")
            elif role == "tool":
                piece = "" if index > 0 and messages[index - 1]["role"] == "tool" else "<|im_start|>user"
                piece += f"\n<tool_response>\n{message['content']}\n</tool_response>"
                if index == len(messages) - 1 or messages[index + 1]["role"] != "tool":
                    piece += "<|im_end|>\n"
                pieces.append(piece)
            else:
                piece = "<|im_start|>assistant\n"
                if self.preserve_thinking or index > last_query_index:
                    piece += f"<think>\n{message.get('reasoning_content', '')}\n</think>\n\n"
                piece += message["content"] + self._render_tool_calls(message)
                pieces.append(piece + "<|im_end|>\n")
        if add_generation_prompt:
            pieces.append("<|im_start|>assistant\n<think>\n" + ("" if enable_thinking else "\n</think>\n\n"))
        return "".join(pieces)

    @staticmethod
    def _render_tool_calls(message):
        pieces = []
        for index, tool_call in enumerate(message.get("tool_calls") or []):
            function_call = tool_call.get("function", tool_call)
            separator = "\n\n" if index == 0 and message["content"].strip() else ("\n" if index else "")
            pieces.append(f"{separator}<tool_call>\n<function={function_call['name']}>\n")
            for argument_name, argument_value in function_call.get("arguments", {}).items():
                pieces.append(f"<parameter={argument_name}>\n{argument_value}\n</parameter>\n")
            pieces.append("</function>\n</tool_call>")
        return "".join(pieces)


def trained(generator, messages):
    token_ids, loss_mask = generator.get_loss_mask(messages)
    assert len(token_ids) == len(loss_mask)
    return generator.get_text_from_loss_mask(token_ids, loss_mask)


TOOL_CALL = {"function": {"name": "terminal", "arguments": {"command": "ls"}}}


def test_thinking_disabled_trains_what_rollout_samples_after_the_generation_prompt():
    tokenizer = Qwen35TemplateTokenizer(preserve_thinking=True)
    generator = MultiTurnLossMaskGenerator(
        tokenizer, tokenizer_type="qwen3_5", chat_template_kwargs={"enable_thinking": False}
    )
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER_1"},
        {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
        {"role": "tool", "content": "README.md"},
        {"role": "assistant", "content": "ANSWER_1"},
        {"role": "user", "content": "USER_2"},
        {"role": "assistant", "content": "ANSWER_2"},
    ]
    token_ids, _ = generator.get_loss_mask(messages)
    assert generator.tokenizer.decode(token_ids) == tokenizer.render(messages, enable_thinking=False)
    # Neither the empty thinking block (part of the generation prompt) nor the newline
    # after <|im_end|> (never sampled) is trained.
    assert trained(generator, messages) == [
        "<tool_call>\n<function=terminal>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call><|im_end|>",
        "ANSWER_1<|im_end|>",
        "ANSWER_2<|im_end|>",
    ]


def test_thinking_enabled_trains_reasoning_and_answer_of_every_turn():
    generator = MultiTurnLossMaskGenerator(Qwen35TemplateTokenizer(preserve_thinking=True), tokenizer_type="qwen3_5")
    messages = [
        {"role": "user", "content": "USER_1"},
        {"role": "assistant", "content": "ANSWER_1", "reasoning_content": "R1"},
        {"role": "user", "content": "USER_2"},
        {"role": "assistant", "content": "ANSWER_2", "reasoning_content": "R2"},
    ]
    assert trained(generator, messages) == [
        "R1\n</think>\n\nANSWER_1<|im_end|>",
        "R2\n</think>\n\nANSWER_2<|im_end|>",
    ]


def test_chat_template_kwargs_reach_the_rendering():
    tokenizer = Qwen35TemplateTokenizer(preserve_thinking=True)
    messages = [{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}]
    for kwargs in ({}, {"enable_thinking": False}):
        generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5", chat_template_kwargs=kwargs)
        assert generator._generation_prompt() == tokenizer.render(
            [{"role": "user", "content": "x"}], add_generation_prompt=True, **kwargs
        )[len(tokenizer.render([{"role": "user", "content": "x"}], **kwargs)) :]
        expected = ["A<|im_end|>"] if kwargs else ["\n</think>\n\nA<|im_end|>"]
        assert trained(generator, messages) == expected


def test_reasoning_rendered_while_thinking_is_disabled_is_refused():
    generator = MultiTurnLossMaskGenerator(
        Qwen35TemplateTokenizer(preserve_thinking=True), tokenizer_type="qwen3_5", chat_template_kwargs={"enable_thinking": False}
    )
    messages = [{"role": "user", "content": "U"}, {"role": "assistant", "content": "A", "reasoning_content": "R"}]
    with pytest.raises(ValueError, match="could not have sampled"):
        generator.get_loss_mask(messages)


def test_a_template_that_drops_earlier_thinking_is_refused_for_those_turns():
    # The model's own template renders the first answer without its thinking block once a
    # later user message exists; rollout never produced that text after its generation prompt.
    generator = MultiTurnLossMaskGenerator(Qwen35TemplateTokenizer(preserve_thinking=False), tokenizer_type="qwen3_5")
    messages = [
        {"role": "user", "content": "USER_1"},
        {"role": "assistant", "content": "ANSWER_1"},
        {"role": "user", "content": "USER_2"},
        {"role": "assistant", "content": "ANSWER_2"},
    ]
    with pytest.raises(ValueError, match="Assistant message 1"):
        generator.get_loss_mask(messages)


def test_step_loss_mask_keeps_a_turn_in_context_without_training_it():
    # An environment that opens with a fixed assistant greeting: the greeting is context the
    # policy never sampled, so the data marks it step_loss_mask 0.
    generator = MultiTurnLossMaskGenerator(
        Qwen35TemplateTokenizer(preserve_thinking=True), tokenizer_type="qwen3_5", chat_template_kwargs={"enable_thinking": False}
    )
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "assistant", "content": "Hi! How can I help you today?", "step_loss_mask": 0},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": "ANSWER"},
    ]
    token_ids, _ = generator.get_loss_mask(messages)
    assert "Hi! How can I help you today?" in generator.tokenizer.decode(token_ids)
    assert trained(generator, messages) == ["ANSWER<|im_end|>"]



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
