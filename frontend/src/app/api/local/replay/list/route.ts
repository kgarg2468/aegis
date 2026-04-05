import { NextResponse } from "next/server";
import { listLocalReplays } from "../../../_lib/runs";

export async function GET() {
  const replays = await listLocalReplays();
  return NextResponse.json({ replays });
}
