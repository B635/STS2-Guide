<script setup>
import { computed, nextTick, ref } from "vue";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const question = ref("");
const loading = ref(false);
const error = ref("");
const useLangGraph = ref(false);
const useReranker = ref(false);
const topN = ref(3);
const candidateN = ref(20);
const messages = ref([
  {
    role: "assistant",
    persist: false,
    content: "你好，我可以基于《杀戮尖塔2》知识库回答问题，并展示 Agent 选择的检索工具、引用来源和校验结果。",
  },
]);
const activeResult = ref(null);
const chatLog = ref(null);

const history = computed(() =>
  messages.value
    .filter((message) => (message.role === "user" || message.role === "assistant") && message.persist !== false)
    .slice(-10)
    .map((message) => ({
      role: message.role,
      content: message.content,
    }))
);

const canSend = computed(() => question.value.trim().length > 0 && !loading.value);

function renderCitations(text) {
  return text
    .replaceAll("[?]", "<span class=\"citation citation-warn\">[?]</span>")
    .replace(/\[(\d+)\]/g, "<span class=\"citation\">[$1]</span>");
}

function formatScore(source) {
  if (typeof source.rerank_score === "number") {
    return `rerank ${source.rerank_score.toFixed(3)}`;
  }
  if (typeof source.score === "number") {
    return `score ${source.score.toFixed(3)}`;
  }
  return "source";
}

async function scrollToBottom() {
  await nextTick();
  if (chatLog.value) {
    chatLog.value.scrollTop = chatLog.value.scrollHeight;
  }
}

async function sendQuestion() {
  const text = question.value.trim();
  if (!text) return;

  error.value = "";
  question.value = "";
  const requestHistory = history.value;
  messages.value.push({ role: "user", content: text });
  await scrollToBottom();

  loading.value = true;
  try {
    const response = await fetch(`${apiBaseUrl}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: text,
        history: requestHistory,
        use_langgraph: useLangGraph.value,
        use_reranker: useReranker.value,
        top_n: topN.value,
        candidate_n: candidateN.value,
      }),
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `请求失败：${response.status}`);
    }

    const payload = await response.json();
    activeResult.value = payload;
    messages.value.push({
      role: "assistant",
      content: payload.answer,
    });
    await scrollToBottom();
  } catch (err) {
    error.value = err.message || "请求失败，请检查 FastAPI 服务是否已启动。";
  } finally {
    loading.value = false;
  }
}

function clearConversation() {
  messages.value = [
    {
      role: "assistant",
      persist: false,
      content: "对话已清空。继续问我卡牌、遗物、怪物、角色或打法问题。",
    },
  ];
  activeResult.value = null;
  error.value = "";
}
</script>

<template>
  <main class="shell">
    <aside class="sidebar">
      <div>
        <p class="eyebrow">STS2 Guide</p>
        <h1>杀戮尖塔2 攻略助手</h1>
      </div>

      <section class="control-group">
        <label class="switch-row">
          <span>LangGraph 工作流</span>
          <input v-model="useLangGraph" type="checkbox" />
        </label>
        <label class="switch-row">
          <span>Reranker 精排</span>
          <input v-model="useReranker" type="checkbox" />
        </label>
      </section>

      <section class="control-group">
        <label class="range-row">
          <span>返回文档数</span>
          <strong>{{ topN }}</strong>
          <input v-model.number="topN" type="range" min="1" max="10" />
        </label>
        <label class="range-row">
          <span>候选池大小</span>
          <strong>{{ candidateN }}</strong>
          <input v-model.number="candidateN" type="range" min="5" max="50" step="5" :disabled="!useReranker" />
        </label>
      </section>

      <button class="secondary-button" type="button" @click="clearConversation">
        清空对话
      </button>

      <section v-if="activeResult" class="trace-panel">
        <div class="panel-heading">
          <span>Agent Trace</span>
          <strong>{{ activeResult.selected_tool }}</strong>
        </div>
        <p class="reason">{{ activeResult.reason }}</p>
        <ol class="trace-list">
          <li v-for="step in activeResult.steps" :key="`${step.tool}-${step.observation}`">
            <span>{{ step.tool }}</span>
            <p>{{ step.observation }}</p>
          </li>
        </ol>
      </section>
    </aside>

    <section class="workspace">
      <div ref="chatLog" class="chat-log">
        <article
          v-for="(message, index) in messages"
          :key="index"
          class="message"
          :class="message.role"
        >
          <div class="role">{{ message.role === "user" ? "You" : "Assistant" }}</div>
          <p v-if="message.role === 'user'">{{ message.content }}</p>
          <p v-else v-html="renderCitations(message.content)"></p>
        </article>
        <article v-if="loading" class="message assistant">
          <div class="role">Assistant</div>
          <p>正在检索、规划工具并生成回答...</p>
        </article>
      </div>

      <div v-if="error" class="error-box">{{ error }}</div>

      <form class="composer" @submit.prevent="sendQuestion">
        <textarea
          v-model="question"
          rows="2"
          placeholder="输入问题，例如：铁甲战士初始血量多少？"
          @keydown.enter.exact.prevent="sendQuestion"
        />
        <button type="submit" :disabled="!canSend">
          发送
        </button>
      </form>

      <section v-if="activeResult" class="evidence">
        <div class="verification" :class="{ passed: activeResult.verification?.passed }">
          <span>Verifier</span>
          <strong>{{ activeResult.verification?.passed ? "passed" : "needs review" }}</strong>
          <p>{{ activeResult.verification?.notes?.join(" ") }}</p>
        </div>

        <div class="sources">
          <article v-for="source in activeResult.sources" :key="source.id" class="source-card">
            <div>
              <strong>[{{ source.id }}]</strong>
              <span>{{ formatScore(source) }}</span>
            </div>
            <p>{{ source.text }}</p>
          </article>
        </div>
      </section>
    </section>
  </main>
</template>
