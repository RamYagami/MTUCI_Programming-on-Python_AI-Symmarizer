const API_BASE = '/api/v1';

const inputArea = document.getElementById('input-text');
const outputArea = document.getElementById('output-area');
const uploadBtn = document.getElementById('upload-btn');
const summarizeBtn = document.getElementById('summarize-btn');

let uploadedFile = null;
let isProcessing = false;

function initCustomSelect(id) {
  const select = document.getElementById(id);
  const selected = select.querySelector('.select-selected');
  const items = select.querySelector('.select-items');
  const allItems = items.querySelectorAll('div');

  selected.addEventListener('click', () => {
    select.classList.toggle('select-open');
  });

  document.addEventListener('click', (e) => {
    if (!select.contains(e.target)) {
      select.classList.remove('select-open');
    }
  });

  allItems.forEach(item => {
    item.addEventListener('click', () => {
      const value = item.getAttribute('data-value');
      selected.setAttribute('data-value', value);
      selected.innerHTML = item.innerHTML;
      select.classList.remove('select-open');
    });
  });
}

initCustomSelect('language-select');
initCustomSelect('length-select');
initCustomSelect('summary-type-select');

uploadBtn.addEventListener('click', () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
      uploadedFile = file;
      inputArea.value = `Загружен файл: ${file.name} (${(file.size / 1024).toFixed(1)} КБ)`;
      inputArea.disabled = true;
    }
  };
  input.click();
});

function getUIConfig() {
  return {
    language: document.querySelector('#language-select .select-selected').getAttribute('data-value'),
    length: document.querySelector('#length-select .select-selected').getAttribute('data-value'),
    mode: document.querySelector('#summary-type-select .select-selected').getAttribute('data-value')
  };
}

async function submitText(text, language, length, mode) {
  const res = await fetch(`${API_BASE}/summarize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, language, length, mode })
  });
  if (!res.ok) throw new Error((await res.json())?.detail?.[0]?.msg || 'Ошибка API');
  return res.json();
}

async function submitFile(file, language, length, mode) {
  const formData = new FormData();
  formData.append('file', file);
  if (language) formData.append('language', language);
  if (length) formData.append('length', length);
  if (mode) formData.append('mode', mode);

  const res = await fetch(`${API_BASE}/summarize/file`, {
    method: 'POST',
    body: formData
  });
  if (!res.ok) throw new Error((await res.json())?.detail?.[0]?.msg || 'Ошибка загрузки файла');
  return res.json();
}

async function pollResult(taskId, onStatusChange) {
  let retries = 0;
  const MAX = 120;
  while (retries++ < MAX) {
    const res = await fetch(`${API_BASE}/result/${taskId}`);
    const data = await res.json();
    onStatusChange?.(data);
    if (data.status === 'finished') return data.summary || '';
    if (data.status === 'failed') throw new Error(data.error || 'Ошибка обработки');
    await new Promise(r => setTimeout(r, 1000));
  }
  throw new Error('Превышено время ожидания');
}

async function summarize() {
  if (isProcessing) return;
  isProcessing = true;
  outputArea.value = 'Подготовка...';

  try {
    const { language, length, mode } = getUIConfig();

    let taskResponse;

    if (uploadedFile) {
      // Отправляем как файл
      outputArea.value = 'Отправка файла на сервер...';
      taskResponse = await submitFile(uploadedFile, language, length, mode);
    } else {
      const text = inputArea.value.trim();
      if (!text) {
        alert('Введите текст или загрузите файл.');
        return;
      }
      if (text.startsWith('Загружен файл:')) {
        alert('Похоже, вы загрузили файл, но он был заменён на текст. Повторите загрузку.');
        return;
      }
      outputArea.value = 'Отправка текста на сервер...';
      taskResponse = await submitText(text, language, length, mode);
    }

    outputArea.value = `Очередь: позиция ${taskResponse.position_in_queue}. Ожидание...`;

    const summary = await pollResult(taskResponse.task_id, (res) => {
      if (res.status === 'queued' && res.position_in_queue) {
        outputArea.value = `Ожидание в очереди (позиция: ${res.position_in_queue})...`;
      } else if (res.status === 'processing') {
        outputArea.value = 'Идёт обработка текста...';
      }
    });

    outputArea.value = summary;
  } catch (err) {
    console.error(err);
    outputArea.value = `Ошибка: ${err.message}`;
    alert(`Ошибка: ${err.message}`);
  } finally {
    isProcessing = false;
    //uploadedFile = null;
    //inputArea.disabled = false;
    //inputArea.value = '';
  }
}

summarizeBtn.addEventListener('click', summarize);
