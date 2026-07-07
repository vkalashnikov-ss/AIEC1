import { initApiPassthrough } from "langgraph-nextjs-api-passthrough";

const apiUrl = process.env.LANGGRAPH_API_URL ?? "http://localhost:2024";
const apiKey = process.env.LANGSMITH_API_KEY || undefined;

export const { GET, POST, PUT, PATCH, DELETE, OPTIONS, runtime } =
  initApiPassthrough({
    apiUrl,
    apiKey,
    runtime: "edge",
    disableWarningLog: true,
  });
