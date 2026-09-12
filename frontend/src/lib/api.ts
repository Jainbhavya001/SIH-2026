import type {
  DocumentType,
  HealthResponse,
  ScanListItem,
  ScanResponse,
  StatsResponse,
} from "@/types";

const BASE = "/api";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore parse errors, fall back to statusText */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  return handle<HealthResponse>(res);
}

export async function getStats(): Promise<StatsResponse> {
  const res = await fetch(`${BASE}/documents/stats`);
  return handle<StatsResponse>(res);
}

export async function listScans(limit = 50): Promise<ScanListItem[]> {
  const res = await fetch(`${BASE}/documents/scans?limit=${limit}`);
  return handle<ScanListItem[]>(res);
}

export async function getScan(id: string): Promise<ScanResponse> {
  const res = await fetch(`${BASE}/documents/scans/${id}`);
  return handle<ScanResponse>(res);
}

export async function scanDocument(params: {
  documentType: DocumentType;
  document: File;
  liveFace?: File | null;
}): Promise<ScanResponse> {
  const form = new FormData();
  form.append("document_type", params.documentType);
  form.append("document", params.document);
  if (params.liveFace) {
    form.append("live_face", params.liveFace);
  }
  const res = await fetch(`${BASE}/documents/scan`, {
    method: "POST",
    body: form,
  });
  return handle<ScanResponse>(res);
}
