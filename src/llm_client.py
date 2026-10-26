"""
Thin provider-agnostic LLM wrapper used by the SciML digest.

Supports Anthropic (Claude), OpenAI and Google Gemini. Gemini is reached through
its OpenAI-compatible endpoint, so only the `anthropic` and `openai` packages
are needed.
"""
import json
import os
import re
import time

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Default (screening, analysis) models per provider. Override in config_sciml.yaml.
DEFAULT_MODELS = {
    "anthropic": ("claude-haiku-5-5", "claude-opus-5-5"),
    "openai": ("gpt-5-nano", "gpt-5-mini"),
    "gemini": ("gemini-2.5-flash", "gemini-2.5-pro"),
}

PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def detect_provider(preferred="auto"):
    """Return the provider to use: the configured one, or the first with an API key set."""
    if preferred and preferred != "auto":
        if preferred not in PROVIDER_ENV_KEYS:
            raise ValueError(f"Unknown provider '{preferred}'. Use one of {list(PROVIDER_ENV_KEYS)} or 'auto'.")
        return preferred
    for provider, env_key in PROVIDER_ENV_KEYS.items():
        if os.environ.get(env_key):
            return provider
    raise RuntimeError(
        "No LLM API key found. Set one of ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY "
        "(e.g. in a .env file)."
    )


class LLMClient:
    def __init__(self, provider, model, reasoning_effort=None):
        self.provider = provider
        self.model = model
        # OpenAI reasoning models (gpt-5*, o*) bill hidden reasoning tokens; a lower effort saves cost.
        self.reasoning_effort = reasoning_effort if re.match(r"(gpt-5|o\d)", model) else None
        self.usage_in = self.usage_out = 0  # token counters, for the cost report
        if provider == "anthropic":
            import anthropic
            self._anthropic = anthropic
            self._client = anthropic.Anthropic(max_retries=5)
        elif provider == "openai":
            import openai
            self._client = openai.OpenAI(max_retries=5)
        elif provider == "gemini":
            import openai
            self._client = openai.OpenAI(
                api_key=os.environ.get("GEMINI_API_KEY"),
                base_url=GEMINI_OPENAI_BASE_URL,
                max_retries=5,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def complete(self, system, prompt, max_tokens=4000, effort=None):
        """Send a single-turn request and return the response text."""
        if self.provider == "anthropic":
            kwargs = {}
            if effort:
                kwargs["output_config"] = {"effort": effort}
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            self.usage_in += response.usage.input_tokens
            self.usage_out += response.usage.output_tokens
            if response.stop_reason == "refusal":
                raise RuntimeError(f"Model refused the request: {response.stop_details}")
            return "".join(block.text for block in response.content if block.type == "text")

        kwargs = {}
        if self.provider == "openai" and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **kwargs,
        )
        self.usage_in += response.usage.prompt_tokens if response.usage else 0
        self.usage_out += response.usage.completion_tokens if response.usage else 0
        if response.choices[0].finish_reason == "length" and not response.choices[0].message.content:
            raise RuntimeError("Ran out of max_completion_tokens during reasoning; raise max_tokens or lower effort")
        return response.choices[0].message.content or ""

    def complete_json(self, system, prompt, max_tokens=4000, effort=None, retries=2):
        """Like `complete`, but parses the first JSON object/array in the reply."""
        last_error = None
        for attempt in range(retries + 1):
            text = self.complete(system, prompt, max_tokens=max_tokens, effort=effort)
            try:
                return extract_json(text)
            except ValueError as e:
                last_error = e
                print(f"  ! Could not parse JSON from {self.model} (attempt {attempt + 1}): {e}")
                time.sleep(2)
        raise last_error


def extract_json(text):
    """Extract the first JSON object or array from an LLM reply (tolerates ``` fences / prose)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    starts = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in response")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[min(starts):])
    return obj
