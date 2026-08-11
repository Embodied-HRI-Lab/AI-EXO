#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 <folder> [commit-message]"
    echo "Example: $0 Higher_Controller/NN_Controller 'Update NN controller'"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 1
fi

target_folder=$1
commit_message=${2:-"Update ${target_folder}"}

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "Error: run this script inside a Git repository." >&2
    exit 1
}

cd "$repo_root"

if [[ ! -d "$target_folder" ]]; then
    echo "Error: folder does not exist: $target_folder" >&2
    exit 1
fi

current_branch=$(git branch --show-current)
if [[ -z "$current_branch" ]]; then
    echo "Error: detached HEAD; check out a branch before pushing." >&2
    exit 1
fi

# Stage only the requested folder and the repository ignore rules.
git add -- "$target_folder"
if [[ -f .gitignore ]]; then
    git add -- .gitignore
fi

if git diff --cached --quiet; then
    echo "Nothing to commit in $target_folder"
    exit 0
fi

echo "Files to commit:"
git diff --cached --stat

git commit -m "$commit_message"
git push origin "$current_branch"

echo "Pushed $target_folder to origin/$current_branch"
