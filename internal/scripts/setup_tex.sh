#!/bin/bash
# Install a user-space TeX Live and the Springer LNCS class for paper builds.
# Usage: bash internal/scripts/setup_tex.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEXROOT=${TEXROOT:-/work/grana_maxillo/lborghi/texlive}
WORK=${TMPDIR:-/tmp}/texlive_install
mkdir -p "$WORK" "$TEXROOT"

if [ ! -x "$TEXROOT/bin/x86_64-linux/pdflatex" ]; then
  cd "$WORK"
  [ -f install-tl-unx.tar.gz ] || curl -sSLO https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
  rm -rf install-tl-*/ && tar xzf install-tl-unx.tar.gz
  cd install-tl-*/
  cat > texlive.profile <<PROF
selected_scheme scheme-small
TEXDIR $TEXROOT
TEXMFCONFIG ~/.texlive/texmf-config
TEXMFHOME ~/texmf
TEXMFLOCAL $TEXROOT/texmf-local
TEXMFSYSCONFIG $TEXROOT/texmf-config
TEXMFSYSVAR $TEXROOT/texmf-var
TEXMFVAR ~/.texlive/texmf-var
option_doc 0
option_src 0
PROF
  ./install-tl -profile texlive.profile
fi

export PATH="$TEXROOT/bin/x86_64-linux:$PATH"
tlmgr option repository https://mirror.ctan.org/systems/texlive/tlnet
tlmgr install booktabs subcaption caption microtype pdfpages xcolor \
  algorithm2e cleveref siunitx multirow makecell orcidlink 2>&1 | tail -5 || true

# Springer LNCS class (CTAN distributes llncs)
mkdir -p "$REPO_DIR/paper"
cd "$WORK"
[ -f llncs.zip ] || curl -sSLO https://mirrors.ctan.org/macros/latex/contrib/llncs.zip
rm -rf llncs && unzip -q -o llncs.zip -d llncs_extract
find llncs_extract -name 'llncs.cls' -exec cp {} "$REPO_DIR/paper/" \;
find llncs_extract -name 'splncs04.bst' -exec cp {} "$REPO_DIR/paper/" \;
ls -la "$REPO_DIR/paper/"
"$TEXROOT/bin/x86_64-linux/pdflatex" --version | head -1
echo "SETUP_TEX_OK"
