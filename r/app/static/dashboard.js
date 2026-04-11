async function apiRequest(path, options = {}) {
    const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
    });

    if (!response.ok) {
        throw new Error(`Request failed: ${path}`);
    }

    return response.json();
}

function formatDate(value) {
    if (!value) {
        return "-";
    }
    return new Date(value).toLocaleString();
}

function renderActiveSightings(items) {
    const container = document.getElementById("active-sightings");
    if (!items.length) {
        container.className = "card-list empty-state";
        container.textContent = "No active detections yet";
        return;
    }

    container.className = "card-list";
    container.innerHTML = items.map((item) => `
        <article class="card-item">
            <div class="card-topline">
                <div class="card-name">${item.label}</div>
                <span class="status-chip ${item.label === "Unknown" ? "unknown" : "active"}">${item.status}</span>
            </div>
            <p class="card-meta">Track #${item.track_id} | Camera: ${item.camera_name}</p>
            <p class="subtle">Confidence ${item.confidence.toFixed(2)} | Movement ${item.movement_score.toFixed(2)}</p>
            <p class="subtle">Last seen ${formatDate(item.last_seen_at)}</p>
        </article>
    `).join("");
}

function renderPersons(items) {
    const container = document.getElementById("persons-list");
    if (!items.length) {
        container.className = "card-list empty-state";
        container.textContent = "No registered persons found";
        return;
    }

    container.className = "card-list";
    container.innerHTML = items.map((item) => `
        <article class="card-item">
            <div class="card-topline">
                <div class="card-name">${item.name}</div>
                <span class="status-chip active">known</span>
            </div>
            <p class="card-meta">${item.sample_count} samples stored in PostgreSQL</p>
            <p class="subtle">Created ${formatDate(item.created_at)}</p>
        </article>
    `).join("");
}

function renderRecent(items) {
    const container = document.getElementById("recent-sightings");
    if (!items.length) {
        container.className = "table-shell empty-state";
        container.textContent = "No sighting history found";
        return;
    }

    container.className = "table-shell";
    container.innerHTML = items.map((item) => `
        <article class="timeline-row">
            <div>
                <div class="card-name">${item.label}</div>
                <p class="timeline-meta">Track #${item.track_id} | ${item.camera_name}</p>
            </div>
            <div>
                <span class="status-chip ${item.status === "exited" ? "exited" : "active"}">${item.status}</span>
            </div>
            <div class="timeline-meta">
                <div>First seen: ${formatDate(item.first_seen_at)}</div>
                <div>Last seen: ${formatDate(item.last_seen_at)}</div>
                <div>Left at: ${formatDate(item.left_at)}</div>
            </div>
        </article>
    `).join("");
}

async function refreshDashboard() {
    try {
        const [health, active, persons, recent] = await Promise.all([
            apiRequest("/health"),
            apiRequest("/sightings/active"),
            apiRequest("/persons"),
            apiRequest("/sightings/recent?limit=20"),
        ]);

        document.getElementById("camera-state").textContent = health.camera_running ? "Running" : "Stopped";
        document.getElementById("active-count").textContent = String(active.length);
        document.getElementById("persons-count").textContent = String(persons.length);
        document.getElementById("recent-count").textContent = String(recent.length);

        renderActiveSightings(active);
        renderPersons(persons);
        renderRecent(recent);
    } catch (error) {
        document.getElementById("camera-state").textContent = "Unavailable";
        console.error(error);
    }
}

async function sendCameraAction(path) {
    try {
        await apiRequest(path, { method: "POST" });
        await refreshDashboard();
    } catch (error) {
        console.error(error);
    }
}

document.getElementById("start-camera").addEventListener("click", () => {
    sendCameraAction("/camera/start");
});

document.getElementById("stop-camera").addEventListener("click", () => {
    sendCameraAction("/camera/stop");
});

refreshDashboard();
setInterval(refreshDashboard, 5000);
