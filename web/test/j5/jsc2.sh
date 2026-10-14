#!/bin/bash
cd "$(dirname "$0")/../../.."
F=""; for n in util markers layout types identity decide fusion revision stream cv aruco cvx fastops ink illum tracks detect recognize scanner; do F="$F web/src/$n.js"; done
S="$1"; shift
exec /System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc $F $S "$@"
