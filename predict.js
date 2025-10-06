document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("prediction-form");
  const outputSection = document.getElementById("prediction-output");
  const resultP = document.getElementById("prediction-result");
  const popupMessage = document.getElementById("popupmessage-prediction");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Mostrar popup
    popupMessage.textContent = "Sending prediction...";
    popupMessage.style.display = "block";
    popupMessage.style.opacity = "1";

    const data = {
      fluorescence: parseFloat(form.fluorescence.value),
      chlorophylle: parseFloat(form.chlorophylle.value),
      absorption: parseFloat(form.absorption.value),
      backscattering: parseFloat(form.backscattering.value),
      particulate_inorganic_carbon: parseFloat(form.particulate_inorganic_carbon.value),
      diffuse_attenuation_coefficient: parseFloat(form.diffuse_attenuation_coefficient.value),
      remote_sensing_reflectance: parseFloat(form.remote_sensing_reflectance.value),
      temperature: parseFloat(form.temperature.value),
    };

    try {
      const response = await fetch("/predict.php", { 
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });

      if (!response.ok) throw new Error("Error: " + response.status);

      const result = await response.json();

      // Redondear hacia arriba y mostrar como porcentaje
      const prob = Math.ceil(result.probabilities * 100);

      resultP.textContent = `Probability of foraging zone: ${prob}%`;
      outputSection.style.display = "block";
      outputSection.style.opacity = "1";
      outputSection.classList.remove("hidden");
      outputSection.classList.add("visible");

    } catch (err) {
      console.error(err);
      resultP.textContent = "An error has occurred";
      outputSection.style.display = "block";
    } finally {
      // Ocultar popup después de mostrar resultado
      setTimeout(() => {
        popupMessage.style.opacity = "0";
        setTimeout(() => { popupMessage.style.display = "none"; }, 300);
      }, 500);
    }
  });
});
