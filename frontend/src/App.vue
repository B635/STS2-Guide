<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref } from "vue";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const mode = ref("chat");
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
const advisorLoading = ref(false);
const advisorError = ref("");
const advisorResult = ref(null);
const realtimeStatus = ref("connecting");
const realtimeMessage = ref("正在连接本地状态桥接…");
const lastRealtimeEventId = ref("");
const advisorForm = ref({
  character: "IRONCLAD",
  ascension: 0,
  act: 1,
  floor: 1,
  hp: null,
  maxHp: null,
  deckText: "",
  relicText: "",
  options: ["", "", ""],
});
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
const canRecommend = computed(
  () =>
    advisorForm.value.options.some((option) => option.trim()) &&
    !advisorLoading.value
);
const realtimeLabel = computed(() => {
  const labels = {
    connecting: "连接中",
    waiting: "等待游戏事件",
    live: "实时建议已更新",
    closed: "选牌界面已关闭",
    stale: "仅发现历史事件",
    offline: "本地桥接未连接",
  };
  return labels[realtimeStatus.value] || "等待游戏事件";
});

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

function parseList(value) {
  return value
    .split(/[\n,，]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseDeck(value) {
  return parseList(value).map((item) => {
    const match = item.match(/^(.*?)(?:\s*[*x×]\s*(\d+))?$/i);
    return {
      card: match?.[1]?.trim() || item,
      count: Number(match?.[2] || 1),
    };
  });
}

async function submitAdvisor() {
  advisorError.value = "";
  advisorResult.value = null;
  advisorLoading.value = true;
  try {
    const state = {
      character: advisorForm.value.character,
      ascension: advisorForm.value.ascension,
      act: advisorForm.value.act,
      floor: advisorForm.value.floor,
      deck: parseDeck(advisorForm.value.deckText),
      relics: parseList(advisorForm.value.relicText),
    };
    if (advisorForm.value.hp !== null && advisorForm.value.hp !== "") {
      state.hp = advisorForm.value.hp;
    }
    if (advisorForm.value.maxHp !== null && advisorForm.value.maxHp !== "") {
      state.max_hp = advisorForm.value.maxHp;
    }

    const response = await fetch(`${apiBaseUrl}/recommend/card-reward`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        state,
        options: advisorForm.value.options
          .map((option) => option.trim())
          .filter(Boolean),
        persist: false,
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `请求失败：${response.status}`);
    }
    advisorResult.value = payload;
  } catch (err) {
    advisorError.value = err.message || "推荐请求失败。";
  } finally {
    advisorLoading.value = false;
  }
}

function useRealtimeAdvice(result) {
  if (!result?.advice) return;
  advisorResult.value = {
    ...result.advice,
    state_id: result.state_id,
    decision_id: result.decision_id,
  };
}

async function pollRealtimeState() {
  if (mode.value !== "advisor") return;
  try {
    const response = await fetch(`${apiBaseUrl}/events/game-state/latest`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    const event = payload.event;
    if (!event) {
      realtimeStatus.value = "waiting";
      realtimeMessage.value = "已连接，等待只读 Mod 事件…";
      return;
    }
    const eventFingerprint = [
      event.event_id,
      event.status,
      event.processed_at || "",
    ].join(":");
    if (eventFingerprint === lastRealtimeEventId.value) return;
    lastRealtimeEventId.value = eventFingerprint;

    const emittedAt = Date.parse(event.emitted_at);
    if (
      Number.isFinite(emittedAt) &&
      Date.now() - emittedAt > 30 * 60 * 1000
    ) {
      realtimeStatus.value = "stale";
      realtimeMessage.value = "桥接已连接，但最近事件超过 30 分钟，未恢复旧建议。";
      return;
    }

    const result = event.result;
    if (event.event_type === "map_choice") {
      realtimeStatus.value = "waiting";
      realtimeMessage.value =
        result?.message || "地图快照已接收，P0 暂不生成路线推荐。";
      return;
    }

    // ── card_reward / decision_closed events ───────────────────────
    if (event.event_type === "decision_closed") {
      realtimeStatus.value = "closed";
      realtimeMessage.value = "已收到关闭事件，旧建议已清除。";
      advisorResult.value = null;
      return;
    }
    if (event.status === "processed" && result?.advice) {
      realtimeStatus.value = "live";
      realtimeMessage.value = `已自动处理事件 ${event.event_id}`;
      useRealtimeAdvice(result);
      return;
    }
    realtimeStatus.value = "waiting";
    realtimeMessage.value = result?.message || "事件已接收，等待推荐器处理。";
  } catch {
    realtimeStatus.value = "offline";
    realtimeMessage.value = "请先启动 FastAPI；Mod 文件桥接可独立运行。";
  }
}

let realtimeTimer;
onMounted(() => {
  pollRealtimeState();
  realtimeTimer = window.setInterval(pollRealtimeState, 1500);
});
onUnmounted(() => {
  window.clearInterval(realtimeTimer);
});
</script>

<template>
  <main class="shell">
    <aside class="sidebar">
      <div>
        <p class="eyebrow">STS2 Guide</p>
        <h1>杀戮尖塔2 攻略助手</h1>
      </div>

      <nav class="mode-switch" aria-label="功能切换">
        <button
          type="button"
          :class="{ active: mode === 'chat' }"
          @click="mode = 'chat'"
        >
          攻略问答
        </button>
        <button
          type="button"
          :class="{ active: mode === 'advisor' }"
          @click="mode = 'advisor'"
        >
          选牌建议
        </button>
      </nav>

      <section v-if="mode === 'chat'" class="control-group">
        <label class="switch-row">
          <span>LangGraph 工作流</span>
          <input v-model="useLangGraph" type="checkbox" />
        </label>
        <label class="switch-row">
          <span>Reranker 精排</span>
          <input v-model="useReranker" type="checkbox" />
        </label>
      </section>

      <section v-if="mode === 'chat'" class="control-group">
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

      <button
        v-if="mode === 'chat'"
        class="secondary-button"
        type="button"
        @click="clearConversation"
      >
        清空对话
      </button>

      <section v-if="mode === 'advisor'" class="control-group advisor-note">
        <strong>P0 本地推荐</strong>
        <p>局面规则优先，社区分数只作小权重先验；不会保存逐次选牌历史。</p>
      </section>

      <section
        v-if="mode === 'advisor'"
        class="control-group realtime-panel"
        :data-status="realtimeStatus"
      >
        <div>
          <span class="status-dot" aria-hidden="true"></span>
          <strong>{{ realtimeLabel }}</strong>
        </div>
        <p>{{ realtimeMessage }}</p>
      </section>

      <section v-if="mode === 'chat' && activeResult" class="trace-panel">
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

    <section v-if="mode === 'chat'" class="workspace">
      <div ref="chatLog" class="chat-log">
        <article
          v-for="(message, index) in messages"
          :key="index"
          class="message"
          :class="message.role"
        >
          <div class="role">{{ message.role === "user" ? "You" : "Assistant" }}</div>
          <p v-if="message.role === 'user'">{{ message.content }}</p>
          <p v-else>{{ message.content }}</p>
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
              <strong>[{{ source.id }}] {{ source.title || source.source_type || "来源" }}</strong>
              <span>{{ formatScore(source) }}</span>
            </div>
            <p v-if="source.author || source.section" class="source-meta">
              {{ [source.author, source.section].filter(Boolean).join(" · ") }}
            </p>
            <p>{{ source.text }}</p>
            <a
              v-if="source.original_url || source.url"
              :href="source.original_url || source.url"
              target="_blank"
              rel="noopener noreferrer"
            >
              查看原始来源
            </a>
          </article>
        </div>
      </section>
    </section>

    <section v-else-if="mode === 'advisor'" class="advisor-workspace">
      <header class="advisor-header">
        <div>
          <p class="eyebrow dark">Live Card Advisor</p>
          <h2>实时选牌建议</h2>
        </div>
        <p>只读 Mod 自动填入；下面的表单保留作调试和无 Mod 降级入口。</p>
      </header>

      <form class="advisor-form" @submit.prevent="submitAdvisor">
        <label>
          <span>角色</span>
          <select v-model="advisorForm.character">
            <option value="IRONCLAD">铁甲战士</option>
            <option value="SILENT">静默猎手</option>
            <option value="DEFECT">故障机器人</option>
            <option value="REGENT">储君</option>
            <option value="NECROBINDER">亡灵契约师</option>
          </select>
        </label>
        <label>
          <span>进阶</span>
          <input v-model.number="advisorForm.ascension" type="number" min="0" />
        </label>
        <label>
          <span>幕</span>
          <input v-model.number="advisorForm.act" type="number" min="1" max="4" />
        </label>
        <label>
          <span>楼层</span>
          <input v-model.number="advisorForm.floor" type="number" min="0" />
        </label>
        <label>
          <span>当前生命</span>
          <input v-model.number="advisorForm.hp" type="number" min="0" placeholder="可选" />
        </label>
        <label>
          <span>最大生命</span>
          <input v-model.number="advisorForm.maxHp" type="number" min="1" placeholder="可选" />
        </label>
        <label class="wide-field">
          <span>当前牌组</span>
          <textarea
            v-model="advisorForm.deckText"
            rows="5"
            placeholder="每行一张，例如：&#10;打击 × 4&#10;防御 × 4&#10;BASH"
          />
        </label>
        <label class="wide-field">
          <span>遗物</span>
          <input
            v-model="advisorForm.relicText"
            placeholder="用逗号分隔，可暂时留空"
          />
        </label>
        <fieldset class="wide-field option-fieldset">
          <legend>奖励候选</legend>
          <input
            v-for="(_, index) in advisorForm.options"
            :key="index"
            v-model="advisorForm.options[index]"
            :placeholder="`候选 ${index + 1}`"
          />
        </fieldset>
        <button class="advisor-submit" type="submit" :disabled="!canRecommend">
          {{ advisorLoading ? "正在分析…" : "生成建议" }}
        </button>
      </form>

      <div v-if="advisorError" class="error-box">{{ advisorError }}</div>

      <section v-if="advisorResult" class="advisor-result">
        <div class="recommendation-summary">
          <div>
            <span>建议</span>
            <strong>
                {{
                  advisorResult.decision_status === "skip"
                    ? "跳过"
                    : advisorResult.decision_status === "uncertain"
                      ? `倾向 ${advisorResult.recommended_option}（证据不足）`
                      : advisorResult.recommended_option
                }}
            </strong>
          </div>
          <div>
            <span>方法</span>
            <strong>{{ advisorResult.method }}</strong>
          </div>
          <div>
            <span>置信度</span>
            <strong>{{ advisorResult.confidence }}</strong>
          </div>
        </div>
        <p class="advisor-disclaimer">{{ advisorResult.disclaimer }}</p>

        <div class="recommendation-grid">
          <article
            v-for="item in advisorResult.recommendations"
            :key="item.option_index"
            class="recommendation-card"
          >
            <header>
              <span>#{{ item.rank }}</span>
              <strong>{{ item.card }}</strong>
              <b>{{ item.score.toFixed(1) }}</b>
            </header>
            <ul>
              <li v-for="factor in item.factors" :key="factor.code">
                <div>
                  <strong>{{ factor.delta >= 0 ? "+" : "" }}{{ factor.delta.toFixed(1) }}</strong>
                  <span>{{ factor.message }}</span>
                </div>
                <a
                  v-if="factor.source_url"
                  :href="factor.source_url"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {{ factor.source_name }} · 样本 {{ factor.sample_size }}
                </a>
              </li>
            </ul>
          </article>
        </div>
      </section>
    </section>
  </main>
</template>
