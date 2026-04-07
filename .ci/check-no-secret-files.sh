#!/usr/bin/env bash
set -euo pipefail

bad_files="$(
  git ls-files | awk '
    {
      n=split($0, parts, "/");
      base=parts[n];
      if (base == "secrets.txt") {
        print $0;
      } else if (base == ".env") {
        print $0;
      } else if (base ~ /^\.env\..+/) {
        print $0;
      }
    }
  '
)"

if [[ -n "${bad_files}" ]]; then
  echo "Blocked secret-like files in repository:"
  echo "${bad_files}"
  exit 1
fi

echo "Secret file guard passed."

