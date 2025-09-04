const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let drawing = false;

canvas.addEventListener("mousedown", () => { drawing = true; ctx.beginPath(); });
canvas.addEventListener("mouseup", () => { drawing = false; });
canvas.addEventListener("mousemove", draw);

function draw(e) {
    if (!drawing) return;
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
}

document.getElementById("predict").addEventListener("click", async () => {
    const dataURL = canvas.toDataURL("image/png");
    console.log("Sending image to Flask...");

    try {
        const res = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataURL }),
        });

        console.log("Fetch status:", res.status);
        const result = await res.json();
        console.log("Response JSON:", result);

        document.getElementById("result").textContent = "Prediction: " + result.prediction;
    } catch (err) {
        console.error("Error during fetch:", err);
        document.getElementById("result").textContent = "Prediction failed.";
    }
});

document.getElementById("clear").addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById("result").textContent = "";
});