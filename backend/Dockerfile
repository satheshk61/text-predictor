# --------------------------
# 1️⃣ Base Image
# --------------------------
FROM python:3.10-slim

# --------------------------
# 2️⃣ Set working directory
# --------------------------
WORKDIR /app

# --------------------------
# 3️⃣ Copy backend files
# --------------------------
COPY . /app

# --------------------------
# 4️⃣ Install dependencies
# --------------------------
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------
# 5️⃣ Expose Flask port
# --------------------------
EXPOSE 5000

# --------------------------
# 6️⃣ Run Flask app
# --------------------------
CMD ["python", "app.py"]
