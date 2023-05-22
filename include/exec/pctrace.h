#ifndef PCTRACE_H
#define PCTRACE_H

#define TB_TRACE_NUM 4096

struct csky_trace_info {
    target_ulong tb_pc;
    bool notjmp;
};

#endif
