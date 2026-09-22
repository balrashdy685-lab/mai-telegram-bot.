!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بوت تلجرام ذكي - يستخدم Gemini و OpenRouter
يرد بأسلوب مباشر ومنسق، وينفذ الطلبات مباشرة بدون تردد أو أسئلة زايدة.
"""

import os
import logging
import asyncio
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import defaultdict

import aiohttp
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ============== الإعدادات ==============
# ضع القيم هنا مباشرة أو عن طريق متغيرات البيئة (الأفضل أمنياً)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "ضع_توكن_البوت_هنا")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "ضع_مفتاح_جيميني_هنا")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "ضع_مفتاح_اوبن_روتر_هنا")

# نماذج مجانية على OpenRouter (يمكن تغييرها إذا صار عندهم تحديث)
OPENROUTER_MODEL = "meta-llama/llama-3.1-405b-instruct:free"
GEMINI_MODEL = "gemini-2.0-flash"

# كم رسالة نحتفظ فيها بالذاكرة لكل مستخدم
MAX_HISTORY = 20

# الشخصية / أسلوب الرد (نفس روح كلود: مباشر، منفذ، منسق)
SYSTEM_PROMPT = """أنت مساعد ذكي في بوت تلجرام. اتبع هذه القواعد بدقة:

1. نفّذ الطلب مباشرة بدون مقدمات مثل "بالتأكيد" أو "يسعدني مساعدتك" - ابدأ بالإجابة نفسها.
2. إذا طلب المستخدم فعل شيء محدد (كتابة كود، صياغة رسالة، تلخيص)، نفّذه فوراً دون طرح أسئلة إضافية إلا إذا كان الطلب غامضاً فعلاً - وحينها افترض المعقول وتابع.
3. نسّق ردودك بـ Markdown يتوافق مع تلجرام:
   - **نص عريض** للعناوين والنقاط المهمة
   - `كود قصير` بين علامتين مائلتين
   - كتل كود بثلاث علامات ```
   - قوائم نقطية أو مرقمة عند الحاجة
4. اجعل الردود مختصرة ومباشرة للأسئلة البسيطة، وأعمق للمواضيع المعقدة - لا تطوّل بلا داعٍ.
5. لا تكرر كلام المستخدم أو تلخص طلبه قبل الإجابة.
6. حافظ على سياق المحادثة السابقة عند الرد.
7. تحدث باللهجة أو اللغة التي يستخدمها المستخدم (عربي فصيح، عامي، أو إنجليزي).
"""

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ذاكرة محادثة بسيطة في الرام لكل مستخدم
user_history = defaultdict(list)


# ============== دوال الاتصال بالـ APIs ==============

async def call_gemini(messages: list) -> str:
    """يرسل الطلب لجيميني ويرجع الرد النصي."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048},
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=60) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"Gemini error {resp.status}: {data}")
            return data["candidates"][0]["content"]["parts"][0]["text"]


async def call_openrouter(messages: list) -> str:
    """يرسل الطلب لـ OpenRouter ويرجع الرد النصي (احتياطي لو جيميني وقع)."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": full_messages,
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=60) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"OpenRouter error {resp.status}: {data}")
            return data["choices"][0]["message"]["content"]


async def get_ai_reply(messages: list) -> str:
    """يحاول جيميني أولاً، ولو فشل يروح على OpenRouter تلقائياً."""
    try:
        return await call_gemini(messages)
    except Exception as e:
        logger.warning(f"Gemini فشل، التحويل لـ OpenRouter: {e}")
        try:
            return await call_openrouter(messages)
        except Exception as e2:
            logger.error(f"OpenRouter فشل أيضاً: {e2}")
            return "⚠️ صار خطأ بالاتصال بالذكاء الاصطناعي. جرب بعد شوي."


# ============== تنسيق الرد لتلجرام (Markdown) ==============

def to_telegram_markdown(text: str) -> str:
    """يحول Markdown العادي لصيغة تلجرام MarkdownV2 بأمان بسيط.
    نستخدم هنا وضع HTML بدل MarkdownV2 لأنه أسهل وأقل عرضة للأخطاء."""
    # تحويل ** bold ** الى <b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # تحويل `code` الى <code>
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    # تحويل كتل الكود ```lang\n...\n```
    text = re.sub(
        r"```(\w+)?\n(.*?)```",
        lambda m: f"<pre><code>{m.group(2)}</code></pre>",
        text,
        flags=re.DOTALL,
    )
    return text


async def send_long_message(update: Update, text: str):
    """تلجرام يوقف الرسايل الطويلة عند 4096 حرف، فنقسمها."""
    formatted = to_telegram_markdown(text)
    chunk_size = 4000
    chunks = [formatted[i:i + chunk_size] for i in range(0, len(formatted), chunk_size)] or [""]

    for chunk in chunks:
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
        except Exception:
            # لو صار خطأ بالتنسيق، نرسل كنص عادي بدون تنسيق
            await update.message.reply_text(re.sub(r"<[^>]+>", "", chunk))


# ============== أوامر البوت ==============

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_history[update.effective_user.id].clear()
    await update.message.reply_text(
        "أهلاً! أنا بوتك الذكي 🤖\n\n"
        "أرسل لي أي سؤال أو طلب وبجاوبك على طول.\n\n"
        "الأوامر المتاحة:\n"
        "/reset - تصفير المحادثة\n"
        "/help - عرض المساعدة"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "بس اكتب سؤالك أو طلبك عادي وأنا بجاوب.\n"
        "أدعم النصوص، الأكواد، والتنسيق.\n"
        "/reset يمسح ذاكرة المحادثة الحالية."
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_history[update.effective_user.id].clear()
    await update.message.reply_text("✅ تم تصفير المحادثة.")


# ============== المعالج الرئيسي للرسائل ==============

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    history = user_history[user_id]
    history.append({"role": "user", "content": user_text})
    history[:] = history[-MAX_HISTORY:]  # قص الذاكرة عشان ما تكبر بلا حدود

    reply = await get_ai_reply(history)

    history.append({"role": "assistant", "content": reply})
    history[:] = history[-MAX_HISTORY:]

    await send_long_message(update, reply)


# ============== سيرفر صحي بسيط (مطلوب لـ Render Web Service المجاني) ==============
# Render (والخدمات المشابهة) تتطلب أن الخدمة تفتح منفذ (PORT) وترد على طلبات HTTP
# عشان تعتبرها "شغالة". هذا سيرفر مصغّر لا علاقة له بمنطق البوت، وظيفته فقط
# يرد "OK" على أي طلب عشان Render يعتبر الخدمة حية.

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running")

    def log_message(self, format, *args):
        pass  # نسكت لوقات السيرفر الصحي عشان ما تزاحم لوقات البوت


def start_health_server():
    port = int(os.getenv("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info(f"🩺 السيرفر الصحي شغّال على المنفذ {port}")


# ============== نقطة التشغيل ==============

def main():
    if "ضع_" in TELEGRAM_TOKEN or "ضع_" in GEMINI_API_KEY:
        print("⚠️  لازم تحط التوكنات والمفاتيح الصحيحة قبل التشغيل (بالأعلى أو كمتغيرات بيئة).")

    start_health_server()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🚀 البوت شغّال...")
    app.run_polling()


if __name__ == "__main__":
    main()
