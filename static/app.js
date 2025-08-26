// Demo texts
const demos = [
  "i am so happy and grateful today!",
  "i feel scared and anxious about tomorrow",
  "this is frustrating, i'm angry at everything",
  "i miss you and love you so much",
  "i'm feeling sad and disappointed",
  "wow, i didn't expect that!"
];

document.getElementById("tryDemo")?.addEventListener("click", () => {
  const t = demos[Math.floor(Math.random() * demos.length)];
  document.getElementById("text").value = t;
});

// batch upload
const uploadBtn = document.getElementById("uploadBtn");
uploadBtn?.addEventListener("click", async () => {
  const fileInput = document.getElementById("fileInput");
  const msg = document.getElementById("batchMsg");
  if (!fileInput.files.length) {
    msg.textContent = "⚠️ Please choose a CSV file.";
    return;
  }
  const form = new FormData();
  form.append("file", fileInput.files[0]);
  msg.textContent = "⏳ Uploading...";
  const res = await fetch("/batch", { method: "POST", body: form });
  const data = await res.json();
  if (data.download) {
    msg.innerHTML = `✅ Processed ${data.rows} rows. <a class="text-yellow-300 underline" href="${data.download}">Download predictions</a>`;
  } else {
    msg.textContent = data.error || "❌ Error processing file.";
  }
});

// Chart.js vibrant colors
if (probaData) {
  const ctx = document.getElementById("probChart");
  const labels = probaData.map(d => d.label);
  const values = probaData.map(d => d.value);
  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Probability",
        data: values,
        backgroundColor: [
          "#f87171", "#facc15", "#34d399", "#60a5fa", "#a78bfa", "#f472b6"
        ],
        borderRadius: 8
      }]
    },
    options: {
      responsive: true,
      animation: {
        duration: 1200,
        easing: "easeOutBounce"
      },
      scales: {
        y: { beginAtZero: true, suggestedMax: 1 }
      }
    }
  });
}
