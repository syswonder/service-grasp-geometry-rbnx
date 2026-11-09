#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
#
# 本包没有自己拉起的外部进程，stop 只做幂等清理（不替代 Driver 的 CMD_SHUTDOWN，
# 相机订阅与 gRPC/MCP 由框架在进程退出时回收）。
set -euo pipefail
rm -f -- rbnx-build/.rbnx-built
exit 0
