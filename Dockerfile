# --------------------------
# 1️⃣ Base Image
# --------------------------
FROM python:3.10-slim

# --------------------------
# 2️⃣ Set working directory
# --------------------------
WORKDIR /app

# --------------------------
# 3️⃣ Copy requirements FIRST for caching
# --------------------------
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------
# 4️⃣ Copy the rest of the backend app
# --------------------------
COPY backend/ /app/

# --------------------------
# 5️⃣ Expose the Hugging Face port
# --------------------------
EXPOSE 7860

# --------------------------
# 6️⃣ Run Flask app
# --------------------------
CMD ["python", "app.py"]
