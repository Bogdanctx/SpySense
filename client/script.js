function getColor(percent) {
    const r = Math.min(255, Math.floor((percent / 100) * 255));
    const g = Math.min(255, Math.floor((1 - percent / 100) * 255));
    return `rgb(${r}, ${g}, 80)`; // from green to red
}

function animateCircle(circleEl, targetPercent) {
    let current = 0;
    const span = circleEl.querySelector('span');

    const step = () => {
        if (current <= targetPercent) {
            span.textContent = `${current}%`;

            const color = getColor(current);
            circleEl.style.background = `conic-gradient(${color} ${current}%, #222 ${current}%)`;

            current++;
            requestAnimationFrame(step);
        }
    };

    step();
}

function createJudgeCircle(label, percent) {
    const judge = document.createElement('div');
    judge.className = 'judge';

    const labelEl = document.createElement('div');
    labelEl.className = 'judge-label';
    labelEl.textContent = label;

    const circle = document.createElement('div');
    circle.className = 'circle';
    circle.innerHTML = `<span>0%</span>`; // starting percent

    judge.appendChild(labelEl);
    judge.appendChild(circle);

    // Animate after DOM is inserted
    setTimeout(() => {
        animateCircle(circle, percent);
    }, 100);

    return judge;
}

async function scan() {
    const input = document.getElementById('fileInput');
    const file = input.files[0];
    const statusDiv = document.getElementById('status');
    const container = document.getElementById('judgesContainer');

    statusDiv.textContent = '';
    container.innerHTML = '';

    if (!file) {
        alert("Please select a file.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    statusDiv.textContent = 'Scanning...';

    try {
        const res = await fetch('http://127.0.0.1:8000/scan', {
            method: 'POST',
            body: formData
        });

        const json = await res.json();

        console.log(json.verdict);

        statusDiv.textContent = 'Scanning finished';
        const verdictDiv = document.getElementById('finalVerdict');
        ensemble_verdict = json.verdict === 1 ? 'Spyware' : 'Not Spyware';
        confidence_score = json.confidence_score;

        if (json.verdict === 0) {
            confidence_score = 1 - confidence_score; // invert for not spyware
        }

        verdictDiv.textContent = `Final Verdict: ${ensemble_verdict} (Confidence Score: ${(confidence_score * 100).toFixed(2)}%)`;

        // Judges
        const scores = Object.values(json.details || {});
        scores.forEach((prob, i) => {
            const percent = Math.round(prob * 100);
            const judge = createJudgeCircle(`Judge #${i + 1}`, percent);
            container.appendChild(judge);
        });

    } catch (err) {
        verdictDiv.textContent = 'Error during scan.';
        console.error(err);
    }
}
