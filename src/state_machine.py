# -*- coding: utf-8 -*-
"""SBR 阶段循环约束状态机：利用工艺循环顺序修正模型逐帧预测。"""
import numpy as np

import config as C


class SBRStateMachine:
    """
    合法转移：阶段 i -> i（保持）或 i -> (i+1) % N（顺序推进，含 排水后间歇->进水 循环）。
    仅当合法候选中概率最高者连续 buffer 帧一致时才执行转移，抑制闪跳。
    """

    def __init__(self, num_classes=C.NUM_CLASSES,
                 transition_buffer=C.SM_TRANSITION_BUFFER, init_state=0):
        self.n = num_classes
        self.current = init_state
        self.buffer = []
        self.transition_buffer = transition_buffer

    def valid_next(self, state):
        return [state, (state + 1) % self.n]

    def update(self, frame_probs):
        """frame_probs: [N] 当前帧类别概率 -> 修正后的阶段索引"""
        valid = self.valid_next(self.current)
        best = valid[int(np.argmax(frame_probs[valid]))]

        self.buffer.append(best)
        if len(self.buffer) > self.transition_buffer:
            self.buffer.pop(0)
        if (len(self.buffer) == self.transition_buffer
                and len(set(self.buffer)) == 1
                and self.buffer[0] != self.current):
            self.current = self.buffer[0]
            self.buffer = []
        return self.current


def smooth_offline(probs, transition_buffer=C.SM_TRANSITION_BUFFER):
    """
    离线整段修正（用于视频文件批处理）：
    先用状态机从视频开头估计初始阶段（取前 2s 概率众数），再顺序推进。
    probs: [T, N] -> stages: [T]
    """
    T, N = probs.shape
    init = int(np.argmax(probs[:max(1, min(4, T))].sum(axis=0)))
    sm = SBRStateMachine(num_classes=N, transition_buffer=transition_buffer,
                         init_state=init)
    return np.array([sm.update(probs[t]) for t in range(T)], dtype=int)
