document.getElementById("predictBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("imageInput");
  const resultDiv = document.getElementById("result");

  if (fileInput.files.length === 0) {
    resultDiv.textContent = "Please upload an image first!";
    return;
  }

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  resultDiv.textContent = "Predicting...";

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    });
    const data = await response.json();
    resultDiv.textContent = `Predicted Text: ${data.prediction}`;
  } catch (error) {
    resultDiv.textContent = "Error predicting text!";
    console.error(error);
  }
});
