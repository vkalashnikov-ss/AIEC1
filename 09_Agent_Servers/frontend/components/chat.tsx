"use client";

import { useEffect, useRef, useState } from "react";
import { useStream } from "@langchain/react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Bot,
  FileText,
  Loader2,
  Search,
  Send,
  User,
  Wrench,
} from "lucide-react";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { getMessageText, toolLabel } from "@/lib/messages";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "/api";

const MARKDOWN_COMPONENTS: Components = {
  h1: ({ ...props }) => (
    <h1 className="mb-2 mt-1 text-base font-semibold tracking-tight text-sky-600" {...props} />
  ),
  h2: ({ ...props }) => (
    <h2 className="mb-2 mt-3 text-sm font-semibold tracking-tight text-emerald-600" {...props} />
  ),
  h3: ({ ...props }) => (
    <h3 className="mb-1 mt-2 text-sm font-medium text-cyan-600" {...props} />
  ),
  p: ({ ...props }) => <p className="my-2 leading-relaxed" {...props} />,
  ul: ({ ...props }) => (
    <ul className="my-2 list-disc space-y-1 pl-5 marker:text-muted-foreground" {...props} />
  ),
  ol: ({ ...props }) => (
    <ol className="my-2 list-decimal space-y-1 pl-5 marker:text-muted-foreground" {...props} />
  ),
  li: ({ ...props }) => <li className="leading-relaxed" {...props} />,
  blockquote: ({ ...props }) => (
    <blockquote
      className="my-3 border-l-2 border-primary/50 bg-primary/5 pl-3 py-1 text-muted-foreground"
      {...props}
    />
  ),
  a: ({ ...props }) => (
    <a
      className="font-medium text-primary underline decoration-primary/50 underline-offset-4 hover:text-primary/80 hover:decoration-primary"
      target="_blank"
      rel="noreferrer"
      {...props}
    />
  ),
  code: ({ className, children, ...props }) => {
    const isBlock = className?.includes("language-");
    if (isBlock) {
      return (
        <code
          className="block overflow-x-auto rounded-md border border-primary/20 bg-primary/10 px-3 py-2 text-xs"
          {...props}
        >
          {children}
        </code>
      );
    }
    return (
      <code className="rounded bg-primary/15 px-1 py-0.5 text-[0.8em] text-primary" {...props}>
        {children}
      </code>
    );
  },
  pre: ({ ...props }) => <pre className="my-3" {...props} />,
  table: ({ ...props }) => (
    <div className="my-3 overflow-x-auto">
      <table className="w-full border-collapse overflow-hidden rounded-md text-xs" {...props} />
    </div>
  ),
  th: ({ ...props }) => (
    <th className="border border-primary/25 bg-primary/15 px-2 py-1 text-left font-medium text-primary" {...props} />
  ),
  td: ({ ...props }) => <td className="border border-primary/20 px-2 py-1" {...props} />,
};

type StreamMessage = ReturnType<typeof useStream>["messages"][number];

const SUGGESTIONS = [
  "How often should I deworm my cat?",
  "What vaccinations do kittens need?",
  "What are signs of feline dehydration?",
];

function toolIcon(name?: string) {
  if (name === "retrieve_information") return <FileText className="size-3" />;
  if (name?.startsWith("tavily")) return <Search className="size-3" />;
  return <Wrench className="size-3" />;
}

export function Chat({ assistantId }: { assistantId: string }) {
  const stream = useStream({ apiUrl: API_URL, assistantId });
  const { messages, isLoading, error } = stream;

  const [input, setInput] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isLoading]);

  const send = (text: string) => {
    const content = text.trim();
    if (!content || isLoading) return;
    stream.submit({ messages: [{ type: "human", content }] });
    setInput("");
  };

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    send(input);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <ScrollArea className="flex-1">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-6">
          {messages.length === 0 && (
            <div className="mt-10 flex flex-col items-center gap-6 text-center">
              <div className="flex size-14 items-center justify-center rounded-full bg-muted">
                <Bot className="size-7 text-muted-foreground" />
              </div>
              <div className="space-y-1">
                <h2 className="text-lg font-medium">Ask the cat health agent</h2>
                <p className="text-sm text-muted-foreground">
                  Streams from your LangGraph deployment via a secure proxy.
                </p>
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <Button
                    key={s}
                    variant="outline"
                    size="sm"
                    onClick={() => send(s)}
                  >
                    {s}
                  </Button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, i) => (
            <MessageRow key={message.id ?? i} message={message} />
          ))}

          {isLoading && <ThinkingRow />}

          {error != null && (
            <Card className="border-destructive/40">
              <CardContent className="text-sm text-destructive">
                {error instanceof Error ? error.message : "Something went wrong."}
              </CardContent>
            </Card>
          )}

          <div ref={endRef} />
        </div>
      </ScrollArea>

      <div className="border-t bg-background">
        <form
          onSubmit={onSubmit}
          className="mx-auto flex w-full max-w-3xl items-center gap-2 px-4 py-3"
        >
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Message the agent..."
            disabled={isLoading}
            className="h-10"
            autoFocus
          />
          <Button
            type="submit"
            size="lg"
            disabled={isLoading || input.trim().length === 0}
            className="h-10"
          >
            {isLoading ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Send className="size-4" />
            )}
          </Button>
        </form>
      </div>
    </div>
  );
}

function MessageRow({ message }: { message: StreamMessage }) {
  const isHuman = message.type === "human";
  const isTool = message.type === "tool";
  const text = getMessageText(message.content);
  const toolCalls =
    message.type === "ai"
      ? (message as unknown as {
          tool_calls?: Array<{ name?: string; id?: string }>;
        }).tool_calls ?? []
      : [];

  if (isTool) {
    return (
      <div className="mx-auto w-full max-w-3xl">
        <details className="group rounded-lg border bg-muted/40 text-sm">
          <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-muted-foreground">
            {toolIcon(message.name)}
            <span className="font-medium text-foreground">
              {toolLabel(message.name)}
            </span>
            <span className="text-xs">tool result</span>
          </summary>
          <pre className="max-h-64 overflow-auto whitespace-pre-wrap px-3 pb-3 text-xs text-muted-foreground">
            {text}
          </pre>
        </details>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex w-full items-start gap-3",
        isHuman && "flex-row-reverse"
      )}
    >
      <Avatar>
        <AvatarFallback>
          {isHuman ? (
            <User className="size-4" />
          ) : (
            <Bot className="size-4" />
          )}
        </AvatarFallback>
      </Avatar>

      <div className={cn("flex max-w-[80%] flex-col gap-2", isHuman && "items-end")}>
        {toolCalls.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {toolCalls.map((tc, idx) => (
              <Badge key={tc.id ?? idx} variant="secondary">
                {toolIcon(tc.name)}
                {toolLabel(tc.name)}
              </Badge>
            ))}
          </div>
        )}

        {text && (
          <div
            className={cn(
              "rounded-2xl px-4 py-2.5 text-sm",
              isHuman
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-foreground"
            )}
          >
            {isHuman ? (
              <div className="whitespace-pre-wrap">{text}</div>
            ) : (
              <div className="max-w-none whitespace-pre-wrap text-sm">
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
                  {text}
                </ReactMarkdown>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ThinkingRow() {
  return (
    <div className="flex w-full items-start gap-3">
      <Avatar>
        <AvatarFallback>
          <Bot className="size-4" />
        </AvatarFallback>
      </Avatar>
      <div className="flex items-center gap-2 rounded-2xl bg-muted px-4 py-3 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Thinking...
      </div>
    </div>
  );
}
