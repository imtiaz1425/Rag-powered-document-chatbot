// Base API URL: relative by default (combined Flask app), or overridden for Netlify deployment
const API_BASE = (typeof window !== 'undefined' && window.API_BASE_URL) ? window.API_BASE_URL.replace(/\/$/, '') : '';

function ts() {
	return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function makeMessageNode(kind, text, pages = [], evidence = []) {
	const wrap = document.createElement('div');
	wrap.className = 'message ' + (kind === 'user' ? 'user' : 'assistant');

	const contentDiv = document.createElement('div');
	contentDiv.className = 'msg-body';
	if (kind === 'assistant' && window.marked) {
		contentDiv.innerHTML = marked.parse(text);
	} else {
		contentDiv.textContent = text;
	}
	wrap.appendChild(contentDiv);

	// Display genuine page citations if present
	if (kind === 'assistant' && pages && pages.length > 0) {
		const citationWrap = document.createElement('div');
		citationWrap.className = 'citation-bar';

		const label = document.createElement('span');
		label.className = 'citation-label';
		label.textContent = 'Sources: ';
		citationWrap.appendChild(label);

		pages.forEach(p => {
			const badge = document.createElement('span');
			badge.className = 'page-pill';
			badge.textContent = `Page ${p}`;
			citationWrap.appendChild(badge);
		});

		wrap.appendChild(citationWrap);
	}

	// Collapsible evidence passages
	if (kind === 'assistant' && evidence && evidence.length > 0) {
		const details = document.createElement('details');
		details.className = 'evidence-accordion';
		const summary = document.createElement('summary');
		summary.textContent = `View Grounded Passages (${evidence.length} chunks retrieved)`;
		details.appendChild(summary);

		const list = document.createElement('div');
		list.className = 'evidence-list';
		evidence.forEach((ev, idx) => {
			const item = document.createElement('div');
			item.className = 'evidence-item';
			item.innerHTML = `<strong>Passage ${idx + 1} (Page ${ev.page} - ${ev.section || 'General'}):</strong><p>${ev.snippet}</p>`;
			list.appendChild(item);
		});
		details.appendChild(list);
		wrap.appendChild(details);
	}

	const t = document.createElement('span');
	t.className = 'ts';
	t.textContent = ts();
	wrap.appendChild(t);

	return wrap;
}

function appendMessage(kind, text, pages = [], evidence = []) {
	const area = document.getElementById('chat-messages');
	const node = makeMessageNode(kind, text, pages, evidence);
	area.appendChild(node);
	area.scrollTop = area.scrollHeight;
	return node;
}

function setPipelineStep(stepId, state = 'done') {
	const el = document.getElementById(stepId);
	if (!el) return;
	el.classList.remove('active', 'done', 'error');
	if (state) el.classList.add(state);
}

function resetPipeline() {
	['step-upload', 'step-extract', 'step-chunk', 'step-index', 'step-ready'].forEach(id => {
		const el = document.getElementById(id);
		if (el) el.classList.remove('active', 'done', 'error');
	});
}

function updateSuggestions(activeFilename) {
	const suggArea = document.getElementById('suggestions');
	suggArea.innerHTML = '';
	const sampleQuestions = [
		'Summarize the key points of this document.',
		'What are the main sections or topics discussed?',
		'What are the primary findings, figures, or data presented?',
		'Explain the core conclusions.'
	];
	sampleQuestions.forEach(q => {
		const chip = document.createElement('div');
		chip.className = 'chip';
		chip.textContent = q;
		suggArea.appendChild(chip);
	});
}

document.addEventListener('DOMContentLoaded', () => {
	const chatInput = document.getElementById('chat-input');
	const sendBtn = document.getElementById('send-btn');
	const uploadBtn = document.getElementById('upload-btn');
	const chatArea = document.getElementById('chat-messages');

	// Initial welcome prompt
	appendMessage('assistant', 'Welcome! Please upload a PDF or Word (.docx) document above to begin asking questions.');

	// Hidden file input
	const hiddenInput = document.createElement('input');
	hiddenInput.type = 'file';
	hiddenInput.accept = '.pdf,.docx';
	hiddenInput.style.display = 'none';
	document.body.appendChild(hiddenInput);

	uploadBtn.addEventListener('click', () => hiddenInput.click());

	// Upload handler
	hiddenInput.addEventListener('change', async (ev) => {
		const file = ev.target.files[0];
		if (!file) return;

		// UI Processing State
		resetPipeline();
		setPipelineStep('step-upload', 'done');
		setPipelineStep('step-extract', 'active');

		document.getElementById('doc-filename').textContent = file.name;
		document.getElementById('doc-type').textContent = file.name.endsWith('.docx') ? 'Word (.docx)' : 'PDF (.pdf)';
		document.getElementById('doc-pages').textContent = 'Processing...';
		document.getElementById('doc-chunks').textContent = 'Processing...';
		document.getElementById('doc-status').textContent = 'Processing...';
		document.getElementById('doc-status').className = 'status-processing';

		document.getElementById('pipeline-message').textContent = `Extracting content and building vector index for ${file.name}...`;
		document.getElementById('rag-banner').className = 'rag-status-banner processing';
		document.getElementById('banner-status-icon').textContent = '⏳';
		document.getElementById('banner-status-text').textContent = `Processing: ${file.name}`;
		document.getElementById('rag-sub').textContent = 'Extracting text and tables, generating embeddings, and building FAISS + BM25 index...';

		chatInput.disabled = true;
		sendBtn.disabled = true;

		const fd = new FormData();
		fd.append('file', file);

		try {
			setPipelineStep('step-chunk', 'active');
			const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: fd });
			const data = await res.json();

			if (!res.ok || data.error) {
				throw new Error(data.error || 'Upload failed');
			}

			// Mark all pipeline stages complete
			['step-upload', 'step-extract', 'step-chunk', 'step-index', 'step-ready'].forEach(id => {
				setPipelineStep(id, 'done');
			});

			// Update Active Document Card
			document.getElementById('doc-filename').textContent = data.filename;
			document.getElementById('doc-type').textContent = (data.file_type || 'PDF').toUpperCase();
			document.getElementById('doc-pages').textContent = data.pages;
			document.getElementById('doc-chunks').textContent = data.chunks;
			document.getElementById('doc-status').textContent = 'Ready';
			document.getElementById('doc-status').className = 'status-ready';

			// Update Banner
			document.getElementById('pipeline-message').textContent = 'RAG Pipeline Ready — All chunks indexed with FAISS & BM25';
			document.getElementById('rag-banner').className = 'rag-status-banner ready';
			document.getElementById('banner-status-icon').textContent = '🟢';
			document.getElementById('banner-status-text').textContent = `DOCUMENT READY: ${data.filename}`;
			document.getElementById('rag-sub').textContent = `Indexed ${data.pages} pages (${data.chunks} chunks). Ask any question grounded in this document.`;

			// System greeting per requirement
			appendMessage('assistant', `Your document is ready! I've processed your document and I'm ready to answer your questions. Ask me anything about it.`);

			// Enable chat
			chatInput.disabled = false;
			sendBtn.disabled = false;
			chatInput.placeholder = `Ask anything about ${data.filename}...`;
			chatInput.focus();

			updateSuggestions(data.filename);

		} catch (err) {
			resetPipeline();
			setPipelineStep('step-upload', 'error');
			document.getElementById('doc-status').textContent = 'Error';
			document.getElementById('doc-status').className = 'status-error';
			document.getElementById('pipeline-message').textContent = `Processing failed: ${err.message}`;
			document.getElementById('rag-banner').className = 'rag-status-banner error';
			document.getElementById('banner-status-icon').textContent = '🔴';
			document.getElementById('banner-status-text').textContent = 'Processing Error';
			document.getElementById('rag-sub').textContent = err.message;

			appendMessage('assistant', `❌ Error processing document: ${err.message}. Please try again with a valid PDF or Word document.`);
		}
		hiddenInput.value = '';
	});

	// Send question handler
	async function handleSend() {
		const text = chatInput.value.trim();
		if (!text) return;

		appendMessage('user', text);
		chatInput.value = '';

		// Add typing indicator
		const typingNode = document.createElement('div');
		typingNode.className = 'message assistant typing-node';
		typingNode.innerHTML = '<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
		chatArea.appendChild(typingNode);
		chatArea.scrollTop = chatArea.scrollHeight;

		try {
			const res = await fetch(`${API_BASE}/ask`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ question: text })
			});
			const data = await res.json();
			typingNode.remove();

			if (!res.ok || data.error) {
				appendMessage('assistant', `⚠️ ${data.error || 'Failed to get an answer.'}`);
			} else {
				appendMessage('assistant', data.answer, data.pages || [], data.evidence || []);
			}
		} catch (err) {
			typingNode.remove();
			appendMessage('assistant', `⚠️ Error communicating with server: ${err.message}`);
		}
	}

	sendBtn.addEventListener('click', handleSend);

	chatInput.addEventListener('keydown', (e) => {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			handleSend();
		}
	});

	// Suggestion chip click
	document.getElementById('suggestions').addEventListener('click', (e) => {
		if (e.target.classList.contains('chip')) {
			chatInput.value = e.target.textContent;
			chatInput.focus();
		}
	});
});
