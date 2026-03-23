const fmt = n => new Intl.NumberFormat('fr-FR',{minimumFractionDigits:2,maximumFractionDigits:2}).format(n);

let allGroups    = [];
let unmatchedRows = [];
let featureLabels = {};
let selectedFourn = 'ALL';

const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', ()=> dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f) uploadFile(f);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) uploadFile(fileInput.files[0]); });

function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  showLoading('Chargement du fichier...');
  fetch('/upload', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      hideLoading();
      if (data.error) { alert('Erreur : ' + data.error); return; }

      document.getElementById('fileInfo').textContent =
        `✓ ${data.source} — ${data.total} écritures chargées`;
      document.getElementById('fileInfo').classList.remove('hidden');
      document.getElementById('runBtn').disabled = false;
      setStatus(`Fichier chargé : ${data.source} · ${data.total} écritures`);

      renderFournisseurs(data.fournisseurs);
      document.getElementById('fournisseurSection').style.display = 'block';
    })
    .catch(() => { hideLoading(); alert('Erreur réseau'); });
}

function renderFournisseurs(list) {
  const el = document.getElementById('fournisseurList');
  el.innerHTML = '';
  const all = makeForurnItem('ALL', list.reduce((s,x)=>s+x.count,0), true);
  el.appendChild(all);
  list.forEach(({fournisseur, count}) => {
    el.appendChild(makeForurnItem(fournisseur, count, false));
  });
}

function makeForurnItem(f, count, active) {
  const d = document.createElement('div');
  d.className = 'fourn-item' + (active ? ' active' : '');
  d.innerHTML = `<span>${f === 'ALL' ? 'Tous fournisseurs' : 'Fourn. ' + f}</span><span class="fourn-count">${count}</span>`;
  d.onclick = () => {
    document.querySelectorAll('.fourn-item').forEach(x => x.classList.remove('active'));
    d.classList.add('active');
    selectedFourn = f;
  };
  return d;
}

function generateAndTrain() {
  showLoading('① Génération dataset synthétique (5 cas CDC)...');
  fetch('/generate-and-train', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      hideLoading();
      const el = document.getElementById('trainResult');
      el.classList.remove('hidden');
      if (data.error) {
        el.className = 'train-result err';
        el.textContent = 'Erreur : ' + data.error;
        return;
      }
      const m = data.meta;
      el.className = 'train-result ok';
      el.innerHTML =
        `✓ Dataset synthétique généré<br>` +
        `✓ Modèle entraîné sur ${m.n_train} paires<br>` +
        `AUC : ${m.auc} · F1 : ${m.f1}<br>` +
        `Positif : ${m.n_positif} · Négatif : ${m.n_negatif}`;

      document.getElementById('modelBadge').innerHTML =
        `<span class="dot green"></span> Modèle ML (synthétique) · AUC ${m.auc} · F1 ${m.f1}`;

      refreshModelTab(m);
    })
    .catch(() => { hideLoading(); alert('Erreur réseau'); });
}

function refreshModelTab(m) {
  if (!m) return;
  const cas_labels = {
    'CAS1_TOLERANCE':     '⚖ Cas 1 — Montants avec tolérance',
    'CAS2_REF_PARTIELLE': '🔗 Cas 2 — Références partielles multi-sources',
    'CAS3_LIBELLE_SIM':   '📝 Cas 3 — Libellés textuels similaires',
    'CAS4_PARTIEL_1N':    '📦 Cas 4 — Paiements partiels 1-N',
    'CAS5_NEGATIF':       '❌ Cas 5 — Cas négatifs',
  };
  const fam_colors = { 'Montants':'00d4ff','Références':'a855f7','Libellé':'00ff9d','Contexte':'ff6b35' };

  let html = `<div class="model-grid">
    <div class="model-kv"><span class="model-key">Algorithme</span><span class="model-val">Random Forest (300 arbres)</span></div>
    <div class="model-kv"><span class="model-key">Données</span><span class="model-val" style="color:var(--accent4)">Dataset synthétique</span></div>
    <div class="model-kv"><span class="model-key">Paires train</span><span class="model-val">${m.n_train}</span></div>
    <div class="model-kv"><span class="model-key">Paires test</span><span class="model-val">${m.n_test}</span></div>
    <div class="model-kv"><span class="model-key">AUC-ROC</span><span class="model-val green">${m.auc}</span></div>
    <div class="model-kv"><span class="model-key">F1-Score</span><span class="model-val green">${m.f1}</span></div>
    <div class="model-kv"><span class="model-key">Précision</span><span class="model-val">${m.precision}</span></div>
    <div class="model-kv"><span class="model-key">Rappel</span><span class="model-val">${m.recall}</span></div>
  </div>`;

  if (m.perf_par_cas) {
    html += `<div class="feat-title" style="margin-top:18px">Performance par cas (CDC 4.2.2)</div>
    <div class="cas-grid">`;
    Object.entries(m.perf_par_cas).forEach(([cas, acc]) => {
      const pct = Math.round(acc * 100);
      html += `<div class="cas-card">
        <div class="cas-label">${cas_labels[cas] || cas}</div>
        <div class="cas-track"><div class="cas-fill" style="width:${pct}%"></div></div>
        <div class="cas-pct">${pct}%</div>
      </div>`;
    });
    html += `</div>`;
  }

  if (m.familles) {
    html += `<div class="feat-title" style="margin-top:18px">Importance par famille (CDC 4.2.3)</div>`;
    Object.entries(m.familles).sort((a,b) => b[1]-a[1]).forEach(([fam, val]) => {
      const pct = Math.round(val * 100);
      const col = fam_colors[fam] || '00d4ff';
      html += `<div class="feat-row">
        <span class="feat-name" style="color:var(--text);font-weight:600">${fam}</span>
        <div class="feat-track"><div class="feat-fill" style="width:${pct}%;background:#${col}"></div></div>
        <span class="feat-pct" style="color:#${col}">${pct}%</span>
      </div>`;
    });
  }

  document.getElementById('modelContent').innerHTML = html;
}

function runEngine() {
  const config = {
    tolerance_abs: parseFloat(document.getElementById('tolAbs').value) || 10,
    tolerance_pct: parseFloat(document.getElementById('tolPct').value) || 0.5,
    rule_ref:  document.getElementById('ruleRef').checked,
    rule_11:   document.getElementById('rule11').checked,
    rule_1n:   document.getElementById('rule1N').checked,
    use_ml:    document.getElementById('useML').checked,
  };
  const ml_threshold = parseFloat(document.getElementById('mlThresh').value) || 0.65;

  showLoading('Moteur de règles en cours...');

  fetch('/run', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ config, fournisseur: selectedFourn, ml_threshold })
  })
  .then(r => r.json())
  .then(data => {
    hideLoading();
    if (data.error) { alert('Erreur : ' + data.error); return; }

    allGroups     = data.groups;
    unmatchedRows = data.unmatched_rows;
    featureLabels = data.feature_labels || {};

    document.getElementById('stTotal').textContent     = data.total;
    document.getElementById('stRule').textContent      = data.rule_groups;
    document.getElementById('stML').textContent        = data.ml_groups;
    document.getElementById('stUnmatched').textContent = data.unmatched;
    document.getElementById('stRate').textContent      = data.rate + '%';

    setStatus(`Rapprochement terminé · ${allGroups.length} groupes · Taux ${data.rate}%`);
    document.getElementById('exportBtn').disabled = false;
    document.getElementById('groupsEmpty').style.display = 'none';

    renderGroups();
    renderUnmatched();
    if (data.diagnostic) renderDiagnostic(data.diagnostic);
    switchTab('groups', document.querySelectorAll('.tab')[0]);
  })
  .catch(() => { hideLoading(); alert('Erreur réseau'); });
}

function renderGroups() {
  const body = document.getElementById('groupsBody');
  body.innerHTML = '';

  if (!allGroups.length) {
    document.getElementById('groupsEmpty').style.display = 'flex';
    return;
  }

  allGroups.forEach(g => {
    const isML   = g.moteur === 'ML';
    const ecartOk = Math.abs(g.ecart) < 0.01;

    let scoreHtml = '—';
    if (g.score !== null && g.score !== undefined) {
      const s   = g.score;
      const col = s >= 0.85 ? '#00ff9d' : s >= 0.7 ? '#f59e0b' : '#ff6b35';
      scoreHtml = `<div class="score-cell">
        <div class="score-track"><div class="score-fill" style="width:${s*100}%;background:${col}"></div></div>
        <span class="score-num" style="color:${col}">${(s*100).toFixed(0)}%</span>
      </div>`;
    }

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color:var(--accent);font-weight:600">${g.id}</td>
      <td>${g.fournisseur}</td>
      <td style="text-align:center">${g.items.length}</td>
      <td>
        <span class="badge ${isML?'badge-ml':'badge-rule'}">${isML?'🤖 ML':'⚙ RÈGLE'}</span>
        <span class="badge badge-type" style="margin-left:4px">${g.match_type}</span>
      </td>
      <td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;color:var(--muted);font-size:11px">${g.regle}</td>
      <td style="text-align:right;font-weight:500">${fmt(g.somme)}</td>
      <td style="text-align:right;color:${ecartOk?'var(--accent2)':'var(--accent3)'};font-weight:600">${fmt(g.ecart)}</td>
      <td>${g.valid ? '<span class="badge badge-ok">✓ ÉQUILIBRÉ</span>' : '<span class="badge badge-warn">⚠ ÉCART</span>'}</td>
      <td><button class="btn-detail" onclick="showDetail('${g.id}')">Détail ▶</button></td>
    `;
    body.appendChild(tr);
  });
}

function renderUnmatched() {
  const body = document.getElementById('unmatchedBody');
  body.innerHTML = '';
  const emptyEl = document.getElementById('unmatchedEmpty');

  if (!unmatchedRows.length) {
    emptyEl.style.display = 'flex';
    return;
  }
  emptyEl.style.display = 'none';

  unmatchedRows.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color:var(--accent)">${r.piece}</td>
      <td>${r.fournisseur}</td>
      <td>${r.date||'—'}</td>
      <td style="text-align:right;color:${r.montant<0?'var(--red)':'var(--green)'}">${fmt(r.montant)}</td>
      <td><span class="badge ${r.dc==='H'?'badge-warn':'badge-ok'}">${r.dc}</span></td>
      <td style="color:var(--muted);max-width:200px;overflow:hidden;text-overflow:ellipsis">${r.texte||'—'}</td>
      <td>${r.type||'—'}</td>
    `;
    body.appendChild(tr);
  });
}

function showDetail(id) {
  const g = allGroups.find(x => x.id === id);
  if (!g) return;

  const isML = g.moteur === 'ML';
  document.getElementById('detailTitle').textContent =
    `${g.id} — ${isML ? '🤖 Rapprochement ML' : '⚙ Rapprochement par Règle'}`;

  let html = '';

  if (g.steps && g.steps.length) {
    html += `<div style="margin-bottom:14px">
      <div class="feat-section-title" style="margin-bottom:8px">📋 Explication — Comment on arrive à ce résultat</div>
      <ul class="steps-list">`;
    g.steps.forEach(s => {
      html += `<li class="${isML?'ml-step':''}">${escHtml(s)}</li>`;
    });
    html += `</ul></div>`;
  }

  if (isML && g.features) {
    html += `<div class="feat-section">
      <div class="feat-section-title">🧮 Variables ML — Contribution de chaque feature</div>`;
    const sorted = Object.entries(g.features).sort((a,b)=>b[1]-a[1]);
    sorted.forEach(([k, v]) => {
      const label = featureLabels[k] || k;
      const pct   = Math.round(v * 100);
      const col   = pct >= 70 ? '#00ff9d' : pct >= 40 ? '#f59e0b' : '#60a5fa';
      html += `<div class="feat-row">
        <span class="feat-name">${label}</span>
        <div class="feat-track"><div class="feat-fill" style="width:${pct}%;background:${col}"></div></div>
        <span class="feat-pct" style="color:${col}">${pct}%</span>
      </div>`;
    });
    html += `</div>`;
  }

  html += `<div class="detail-items">
    <div class="detail-items-title">📄 Écritures du groupe (${g.items.length})</div>
    <div class="table-wrap">
    <table><thead><tr>
      <th>N° Pièce</th><th>Date</th><th>Montant</th><th>D/C</th><th>Texte</th><th>Type</th><th>Référence</th>
    </tr></thead><tbody>`;
  g.items.forEach(r => {
    html += `<tr>
      <td style="color:var(--accent)">${r.piece}</td>
      <td>${r.date||'—'}</td>
      <td style="text-align:right;color:${r.montant<0?'var(--red)':'var(--green)'};font-weight:500">${fmt(r.montant)}</td>
      <td><span class="badge ${r.dc==='H'?'badge-warn':'badge-ok'}">${r.dc}</span></td>
      <td style="color:var(--muted);max-width:200px;overflow:hidden;text-overflow:ellipsis">${escHtml(r.texte||'—')}</td>
      <td>${r.type||'—'}</td>
      <td style="color:var(--accent4)">${r.rapproch||'—'}</td>
    </tr>`;
  });
  html += `</tbody></table></div></div>`;

  document.getElementById('detailBody').innerHTML = html;
  const panel = document.getElementById('detailPanel');
  panel.style.display = 'block';
  panel.scrollIntoView({behavior:'smooth', block:'nearest'});
}

function closeDetail() {
  document.getElementById('detailPanel').style.display = 'none';
}

function exportExcel() {
  window.location.href = '/export';
}

function switchTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  if (el) el.classList.add('active');
  ['groups','unmatched','diagnostic','model'].forEach(t => {
    document.getElementById('tab-'+t).style.display = t === name ? 'block' : 'none';
  });
}

function renderDiagnostic(diag) {
  const el = document.getElementById('diagContent');
  if (!diag || !diag.categories || diag.categories.length === 0) {
    el.innerHTML = '<div class="diag-empty">✅ Toutes les écritures ont été rapprochées !</div>';
    document.querySelectorAll('.tab')[2].textContent = '🔍 Diagnostic 0%';
    return;
  }

  const pct = diag.taux_non_resolus;
  document.querySelectorAll('.tab')[2].textContent = `🔍 Diagnostic ${pct}%`;

  const colorsMap = {
    orange: '#ff8c42', red: '#f87171', purple: '#a855f7',
    yellow: '#fbbf24', blue: '#00b4d8'
  };

  let html = `<div class="diag-header">
    <div class="diag-kpi">
      <div class="diag-kpi-val" style="color:var(--accent4)">${diag.total_resolus}</div>
      <div class="diag-kpi-label">Écritures rapprochées</div>
    </div>
    <div class="diag-kpi">
      <div class="diag-kpi-val" style="color:#f87171">${diag.total_non_resolus}</div>
      <div class="diag-kpi-label">Non résolues</div>
    </div>
    <div class="diag-kpi">
      <div class="diag-kpi-val" style="color:var(--accent4)">${diag.taux_resolus}%</div>
      <div class="diag-kpi-label">Taux rapprochement</div>
    </div>
    <div class="diag-kpi">
      <div class="diag-kpi-val" style="color:#f87171">${diag.taux_non_resolus}%</div>
      <div class="diag-kpi-label">Non résolu</div>
    </div>
  </div>`;

  html += `<div style="display:flex;height:12px;border-radius:6px;overflow:hidden;margin-bottom:16px;gap:1px">`;
  html += `<div style="flex:${diag.taux_resolus};background:var(--accent4)" title="Rapproché ${diag.taux_resolus}%"></div>`;
  diag.categories.forEach(cat => {
    const col = colorsMap[cat.color] || '#888';
    html += `<div style="flex:${cat.pct_total};background:${col}" title="${cat.label} ${cat.pct_total}%"></div>`;
  });
  html += `</div>`;

  html += `<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;font-size:10px">`;
  html += `<span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:2px;background:var(--accent4);display:inline-block"></span>Rapproché ${diag.taux_resolus}%</span>`;
  diag.categories.forEach(cat => {
    const col = colorsMap[cat.color] || '#888';
    html += `<span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:2px;background:${col};display:inline-block"></span>${cat.label} ${cat.pct_total}%</span>`;
  });
  html += `</div>`;

  html += `<div class="diag-cats">`;
  diag.categories.forEach((cat, ci) => {
    const col = colorsMap[cat.color] || '#888';
    html += `
    <div class="diag-cat">
      <div class="diag-cat-header" onclick="toggleDiagCat(${ci})">
        <div class="diag-cat-dot" style="background:${col}"></div>
        <div class="diag-cat-title">${cat.label}</div>
        <div class="diag-cat-badges">
          <span class="diag-badge" style="background:${col}22;color:${col}">${cat.n_groupes} groupe${cat.n_groupes>1?'s':''}</span>
          <span class="diag-badge" style="background:${col}22;color:${col}">${cat.n_ecritures} écriture${cat.n_ecritures>1?'s':''}</span>
          <span class="diag-badge" style="background:${col}33;color:${col};font-size:11px">${cat.pct_total}%</span>
        </div>
        <span class="diag-cat-arrow" id="diag-arrow-${ci}">▶</span>
      </div>
      <div class="diag-cat-body" id="diag-body-${ci}">
        <div class="diag-cause">📋 <strong>Cause :</strong> ${cat.cause}</div>
        <div class="diag-solution">💡 <strong>Solution :</strong> ${cat.solution}</div>
        <div class="diag-grp-list">`;

    const maxGrps = Math.min(cat.groupes.length, 10);
    for (let gi = 0; gi < maxGrps; gi++) {
      const g = cat.groupes[gi];
      html += `<div class="diag-grp">
        <div class="diag-grp-title">
          📌 Réf. ${g.ref} &nbsp;|&nbsp; ${g.n} écriture${g.n>1?'s':''} &nbsp;|&nbsp;
          Résidu : <span style="color:${g.somme>=0?'#f87171':'#fb923c'}">${g.somme>=0?'+':''}${g.somme.toLocaleString('fr-FR')} XOF</span>
          ${g.pct_ecart > 0 ? `&nbsp;|&nbsp; ${g.pct_ecart}%` : ''}
          ${Object.keys(g.devises).length > 1 ? `&nbsp;|&nbsp; <span style="color:#a855f7">${Object.keys(g.devises).join(' + ')}</span>` : ''}
        </div>
        <table class="diag-grp-ecr">
          <tr><th>N° Pièce</th><th>Montant</th><th>Devise</th><th>D/C</th><th>Type</th><th>Texte</th></tr>`;
      g.ecritures.forEach(e => {
        const cls = e.montant >= 0 ? 'pos' : 'neg';
        const sign = e.montant >= 0 ? '+' : '';
        html += `<tr>
          <td>${e.piece}</td>
          <td class="${cls}">${sign}${e.montant.toLocaleString('fr-FR')}</td>
          <td>${e.devise}</td>
          <td>${e.dc}</td>
          <td>${e.type}</td>
          <td style="color:var(--muted)">${e.texte || '—'}</td>
        </tr>`;
      });
      html += `</table></div>`;
    }

    if (cat.groupes.length > 10) {
      html += `<div class="diag-more">... et ${cat.groupes.length - 10} autres groupes</div>`;
    }

    html += `</div></div></div>`;
  });

  html += `</div>`;
  el.innerHTML = html;
}

function toggleDiagCat(ci) {
  const body  = document.getElementById(`diag-body-${ci}`);
  const arrow = document.getElementById(`diag-arrow-${ci}`);
  const open  = body.classList.toggle('open');
  arrow.classList.toggle('open', open);
}

function setStatus(msg) {
  document.getElementById('statusText').textContent = msg;
}
function showLoading(msg) {
  document.getElementById('loadingText').textContent = msg || 'Traitement...';
  document.getElementById('loading').style.display = 'flex';
}
function hideLoading() {
  document.getElementById('loading').style.display = 'none';
}
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}