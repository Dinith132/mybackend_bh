#!/usr/bin/env bash
# Resolves a TensorFlow-compatible Python (3.12 or 3.11).
# Source this script, then use $PYTHON_BIN.
#
# Optional env:
#   PYTHON_VERSION=3.12|3.11  (default: try 3.12 then 3.11)
#   INSTALL_PYTHON=1          (default: 1 on bootstrap, set 0 to only resolve)

resolve_python_bin() {
  local version="$1"
  if command -v "python${version}" >/dev/null 2>&1; then
    echo "python${version}"
    return 0
  fi
  return 1
}

find_existing_python() {
  local preferred="${PYTHON_VERSION:-}"

  if [ -n "$preferred" ]; then
    resolve_python_bin "$preferred" && return 0
    return 1
  fi

  resolve_python_bin 3.12 || resolve_python_bin 3.11
}

install_python_from_apt() {
  local version="${1:-3.12}"

  echo "==> Installing Python ${version} via apt"
  sudo apt update
  sudo apt install -y software-properties-common

  # deadsnakes provides 3.11/3.12 on Ubuntu 22.04; safe to try on other releases
  sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
  sudo apt update

  sudo apt install -y \
    "python${version}" \
    "python${version}-venv" \
    "python${version}-dev"
}

install_python_build_deps() {
  sudo apt update
  sudo apt install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    git \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libffi-dev \
    liblzma-dev
}

install_python_with_pyenv() {
  local version="${1:-3.12}"
  local patch_version=""

  case "$version" in
    3.12) patch_version="3.12.8" ;;
    3.11) patch_version="3.11.11" ;;
    *) patch_version="${version}.8" ;;
  esac

  echo "==> Installing Python ${patch_version} via pyenv"
  install_python_build_deps

  if ! command -v pyenv >/dev/null 2>&1; then
    curl -fsSL https://pyenv.run | bash
    export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
  fi

  pyenv install -s "$patch_version"
  echo "$PYENV_ROOT/versions/${patch_version}/bin/python"
}

ensure_python() {
  local bin=""
  local versions_to_try=()

  if [ -n "${PYTHON_VERSION:-}" ]; then
    versions_to_try=("$PYTHON_VERSION")
  else
    versions_to_try=(3.12 3.11)
  fi

  bin="$(find_existing_python || true)"
  if [ -n "$bin" ]; then
    PYTHON_BIN="$bin"
    echo "==> Using existing $("$PYTHON_BIN" --version)"
    return 0
  fi

  if [ "${INSTALL_PYTHON:-1}" != "1" ]; then
    echo "ERROR: No compatible Python found (need 3.12 or 3.11). Set INSTALL_PYTHON=1 to install."
    return 1
  fi

  for version in "${versions_to_try[@]}"; do
    echo "==> Python ${version} not found, attempting install"
    if install_python_from_apt "$version" 2>/dev/null; then
      bin="$(resolve_python_bin "$version" || true)"
      if [ -n "$bin" ]; then
        PYTHON_BIN="$bin"
        echo "==> Installed $("$PYTHON_BIN" --version) via apt"
        return 0
      fi
    fi

    echo "==> apt install failed for Python ${version}, trying pyenv"
  done

  PYTHON_BIN="$(install_python_with_pyenv "${versions_to_try[0]}")"
  echo "==> Installed $("$PYTHON_BIN" --version) via pyenv"
}
