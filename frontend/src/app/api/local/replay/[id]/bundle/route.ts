import { NextResponse } from "next/server";
import { loadLocalReplayBundle } from "../../../../_lib/runs";

type RouteParams = {
  params: Promise<{ id: string }>;
};

export async function GET(_request: Request, context: RouteParams) {
  const { id } = await context.params;
  const bundle = await loadLocalReplayBundle(id);

  if (!bundle) {
    return NextResponse.json({ error: "Replay bundle not found" }, { status: 404 });
  }

  return NextResponse.json(bundle);
}
