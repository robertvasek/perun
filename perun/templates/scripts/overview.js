document.querySelectorAll(".slider-container").forEach(container => {
    const sliderIndex = parseFloat(container.dataset.sliderIndex);
    const arrow = container.querySelector(".arrow-indicator");
    const segmentsWrapper = container.querySelector(".segments-wrapper");

    const maxSegments = 9;
    const clampedIndex = Math.max(0, Math.min(maxSegments, sliderIndex));
    const percentage = (clampedIndex / maxSegments) * 100;

    const wrapperRect = segmentsWrapper.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    const top = wrapperRect.top - containerRect.top - (arrow.offsetHeight || 20);
    arrow.style.top = `${top}px`;

    arrow.style.left = `calc(${percentage}% - 0.5em)`;
});
