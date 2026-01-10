const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : '/api';

let token = localStorage.getItem('token');

export function getToken() {
  return token;
}

export function setToken(t) {
  token = t;
  if (t) {
    localStorage.setItem('token', t);
  } else {
    localStorage.removeItem('token');
  }
}

export function getUser() {
  const u = localStorage.getItem('user');
  return u ? JSON.parse(u) : null;
}

export function setUser(u) {
  if (u) {
    localStorage.setItem('user', JSON.stringify(u));
  } else {
    localStorage.removeItem('user');
  }
}

export async function api(endpoint, method = 'GET', body = null) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const options = { method, headers };
  if (body) {
    options.body = JSON.stringify(body);
  }

  const response = await fetch(`${API_BASE}${endpoint}`, options);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || 'API Error');
  }

  return data;
}

// Auth
export async function login(username, password) {
  const data = await api('/auth/login', 'POST', { username, password });
  setToken(data.access_token);
  setUser(data.user);
  return data;
}

export async function register(username, password) {
  const data = await api('/auth/register', 'POST', { username, password });
  setToken(data.access_token);
  setUser(data.user);
  return data;
}

export function logout() {
  setToken(null);
  setUser(null);
}

// Sessions
export const startSession = (mode) => api('/sessions/start', 'POST', { mode });
export const sendMessage = (sessionId, message) => api(`/sessions/${sessionId}/message`, 'POST', { message });
export const endSession = (sessionId) => api(`/sessions/${sessionId}/end`, 'POST');
export const pauseSession = (sessionId) => api(`/sessions/${sessionId}/pause`, 'POST');
export const resumeSession = (sessionId) => api(`/sessions/${sessionId}/resume`, 'POST');
export const deleteSession = (sessionId) => api(`/sessions/${sessionId}`, 'DELETE');
export const getHistory = () => api('/sessions/history');

// Profile
export const getProfile = () => api('/evolution/profile');
export const getEvolution = (archetype) => api(`/evolution/archetype/${archetype}`);

// Info
export const getInfo = () => api('/info');

// Corpus/Books
export const getCorpusBooks = () => api('/corpus/books');
export const processBook = (text, title, author) => api('/corpus/process', 'POST', { text, title, author });
