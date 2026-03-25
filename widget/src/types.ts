export interface WidgetConfig {
  apiKey: string;
  apiUrl: string;
  theme: "light" | "dark";
  accentColor: string;
  position: "bottom-right" | "bottom-left";
  placeholder: string;
}

export interface SourceChunk {
  title: string;
  url: string | null;
  snippet: string;
  chunk_id: string;
}

export interface WidgetResponse {
  answer: string;
  sources: SourceChunk[];
  verified: boolean;
  confidence: number;
  query_id: string;
}

export interface StreamEvent {
  type: "chunk" | "source" | "verified" | "done";
  data: string | SourceChunk | { verified: boolean; confidence: number };
}
