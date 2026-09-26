"""翻页点规划动态规划核心。

问题建模
========
小节按顺序编号 0..n-1，每小节高度 ``heights[i]``，第 i 小节之后的静默时间为
``silences[i]``（单位：秒）。一页可容纳连续若干小节，且总高度不超过
``page_height``。除最后一页外，每个断页位置 b（表示第 b-1 小节之后翻页，
即下一页从第 b 小节开始）必须满足 ``silences[b-1] >= min_turn_time``。

目标（按优先级）：
1. 最小化总页数；
2. 在页数最少的方案中，最大化所有翻页点静默时间的最小值；
3. 仍并列时，选择断页序号序列字典序最小的方案。

三个目标均在**全篇统一比较**：必翻/禁翻约束只改变合法断页位置集合，
不会把乐谱按必翻点切成若干段各自取局部最优。

可选约束（``required_breaks`` / ``forbidden_breaks``，元素为新页起始小节的
0 基序号，合法范围 1..n-1）：
- 必翻位置一定是断页（其静默还必须达到最低翻页时间，否则无解）；
- 禁翻位置一定不是断页；
- 两数组各自重复、互相交叉，或含 0、末尾序号 n 及越界值，均判定为输入非法。

求解分三步（均为动态规划/基于 DP 的贪心重构）：
1. 前向最短路 DP 求最小页数 K；必翻点通过页首下界强制断页、禁翻点从允许
   位置集合剔除；不可达即无合法方案；
2. 对"所有翻页点静默 >= t"这一可行性判定做二分（判定仍是同一条最短路 DP，
   只是允许的断页位置减少），求最大可行 t；
3. 在阈值 t 下做后缀 min/max 页数 DP（必翻点通过页尾上界强制），贪心选择
   最小的合法断页序号，得到字典序最小的断页序列。

数值表示
========
所有输入数值（小节高度、静默秒数、页面可用高度、最低翻页时间）先按用户书写
的**十进制字面量**解释（如 0.1 就是精确的 1/10，经 ``Decimal(str(x))`` 转换），
再按"高度与页高"、"静默与最低翻页时间"两组分别放大为**同比例整数**（定点
十进制：比例取组内最大的小数位数，乘 10**k 后每个值都是整数）；累计高度为
整数前缀和。单节容量、整页容量、最低翻页静默、最短静默阈值搜索与断页重构
中的全部比较因此都是**精确的十进制比较**，不使用浮点容差。因此：

- 略超边界即不合格：高度 1.0000000005 对页高 1 判定单节超高；
  静默 0.9999999999 对最低翻页时间 1 的位置不可翻页；
- 恰好达到边界仍合格：0.1 + 0.2 在十进制下恰为 0.3，可同页排入页高 0.3；
  静默恰好等于最低翻页时间（或阈值）允许翻页（>= 判定）；
- 输入数值必须是有限值（不接受 NaN 或 Infinity），否则直接报错。

单小节高度超过页面可用高度时无法排版，直接报错。
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, List, Optional, Sequence

INF = 10**15


def _to_scaled_ints(values: Sequence[float]) -> tuple[List[int], int]:
    """把一组十进制字面量转成同一比例下的整数，返回 ``(整数列表, 比例 10**k)``。

    每个值先按 ``str`` 的最短十进制表示（即用户书写的数值）转成 ``Decimal``
    ——0.1 即精确的 1/10，不带二进制浮点表示误差；k 取所有值小数位数的最大
    值，使每个值乘 10**k 后都是整数。同组数值在该比例下做整数精确比较，
    0.1 + 0.2 与 0.3 因此严格相等。
    """
    decs = [Decimal(str(v)) for v in values]
    k = 0
    for d in decs:
        exp = d.as_tuple().exponent
        if exp < 0 and -exp > k:
            k = -exp
    if k:
        return [int(d.scaleb(k)) for d in decs], 10 ** k
    return [int(d) for d in decs], 1


class PlanningError(ValueError):
    """规划失败：输入语义不合法（单小节超高、无合法方案等）。"""


@dataclass(frozen=True)
class Page:
    """一页的排版结果。小节序号均为 0 基、端点闭区间。"""

    page_index: int
    start_measure: int  # 起始小节（0 基）
    end_measure: int  # 结束小节（0 基，闭区间）
    measure_count: int
    used_height: float
    # 本页之后翻页点的静默秒数；最后一页无翻页点，为 None
    turn_silence: Optional[float]


@dataclass(frozen=True)
class Plan:
    pages: List[Page]
    page_count: int
    # 所有翻页点中最短的静默秒数；单页（无翻页点）时为 None
    shortest_turn_silence: Optional[float]
    # 断页序号序列：每个元素是"新一页起始小节"的 0 基序号
    breaks: List[int]


def _page_end_range(prefix: Sequence[int], r: int, page_height: int) -> tuple[int, int]:
    """以第 r 个小节（0 基）为页尾时，页首 l（0 基）的取值范围。

    页内容为小节 l..r，高度为 prefix[r+1] - prefix[l] <= page_height，
    即 prefix[l] >= prefix[r+1] - page_height；整数精确比较下
    bisect_left 直接给出最小的合法 l。
    """
    threshold = prefix[r + 1] - page_height
    lo = bisect_left(prefix, threshold, 0, r + 1)
    return lo, r


def _min_pages_forward(
    n: int,
    prefix: Sequence[int],
    page_height: int,
    allowed: Callable[[int], bool],
    floor_forced: Optional[Sequence[int]] = None,
) -> List[int]:
    """前向 DP：f[r] = 覆盖小节 0..r 所需的最少页数。

    一页小节 l..r 合法当且仅当高度不超限，且 l == 0 或断页位置 l 允许翻页。
    滑动窗口（l 的可行区间随 r 单调）用单调队列维护 f[l-1] 的最小值，
    整体 O(n)。

    ``floor_forced[r]`` 给出以 r 为页尾时页首 l 的下界：<= r 的最后一个必翻点
    （0 表示无）。必翻点 q 必须恰好是某页页首，因此含 q 的页只能以 q 起始，
    强制 l >= floor_forced[r]。
    """
    if floor_forced is None:
        floor_forced = [0] * n
    # 状态以"已覆盖小节数 m"表示：g[m] = f[m-1]，g[0] = 0（空前缀）。
    g: List[int] = [INF] * (n + 1)
    g[0] = 0
    dq: deque[int] = deque()
    for r in range(n):
        m = r  # 新候选：页 [r..r]，对应前缀状态 g[r]
        # 在 r 处断页意味着下一页从第 r 小节开始，r>0 时必须允许翻页。
        if g[m] < INF and (m == 0 or allowed(m)):
            while dq and g[dq[-1]] >= g[m]:
                dq.pop()
            dq.append(m)
        lo, _ = _page_end_range(prefix, r, page_height)
        # 必翻点下界同样可能高于高度窗口下界，取两者较大者。
        lo = max(lo, floor_forced[r])
        while dq and dq[0] < lo:
            dq.popleft()
        g[r + 1] = g[dq[0]] + 1 if dq else INF
    return [g[r + 1] for r in range(n)]


def _suffix_min_max_pages(
    n: int,
    prefix: Sequence[int],
    page_height: int,
    allowed: Callable[[int], bool],
    next_forced: Optional[Sequence[int]] = None,
) -> tuple[List[int], List[int]]:
    """后缀 DP：从第 i 小节开始排完剩余内容所需的最少/最多页数。

    允许页数在 [smin[i], smax[i]] 之间连续可取（任意合法方案都能在允许翻页的
    位置把一页继续拆成两页，从而逐页 +1，故中间页数均可达；必翻点只是把部分
    拆分位置改成强制，区间连续性不受影响）。
    窗口随 i 反向单调，用两个单调队列分别维护最小、最大值，整体 O(n)。

    ``next_forced[i]`` 为严格大于 i 的第一个必翻点（不存在时为 n）：以 i 为
    页首的页不能越过该必翻点，故页尾上界 e <= next_forced[i]，必翻点因此
    强制成为下一页页首。
    """
    if next_forced is None:
        next_forced = [n] * (n + 1)
    smin: List[int] = [INF] * (n + 1)
    smax: List[int] = [-INF] * (n + 1)
    smin[n] = 0
    smax[n] = 0

    min_dq: deque[int] = deque()  # 按 smin 值升序
    max_dq: deque[int] = deque()  # 按 smax 值降序

    for i in range(n - 1, -1, -1):
        # 上一轮窗口里 e == i 的候选已过期（页必须至少包含第 i 小节，e > i）。
        # 索引从右（最新、最小）加入，故从右端剔除 e <= i。
        while min_dq and min_dq[-1] <= i:
            min_dq.pop()
        while max_dq and max_dq[-1] <= i:
            max_dq.pop()
        # 新候选 e = i+1：页 [i..i] 后接后缀 i+1。断页位置 i+1 必须允许翻页，
        # 除非 i+1 == n（本页即最后一页）。e = i+1 必不越过下一个必翻点。
        e = i + 1
        if e == n or allowed(e):
            if smin[e] < INF:
                while min_dq and smin[min_dq[-1]] >= smin[e]:
                    min_dq.pop()
                min_dq.append(e)
            if smax[e] > -INF:
                while max_dq and smax[max_dq[-1]] <= smax[e]:
                    max_dq.pop()
                max_dq.append(e)
        # 页 [i..e-1] 高度 = prefix[e] - prefix[i] <= page_height，
        # 即 prefix[e] <= prefix[i] + page_height；整数精确比较下
        # bisect_right 给出最后一个不超限元素的后一位置，减 1 得最大合法 e。
        max_height = prefix[i] + page_height
        max_e = bisect_right(prefix, max_height, i + 1, n + 1) - 1
        # 必翻点上界：本页不得越过 i 之后的第一个必翻点。
        max_e = min(max_e, next_forced[i])
        while min_dq and min_dq[0] > max_e:
            min_dq.popleft()
        while max_dq and max_dq[0] > max_e:
            max_dq.popleft()
        if min_dq:
            smin[i] = smin[min_dq[0]] + 1
        if max_dq:
            smax[i] = smax[max_dq[0]] + 1
    return smin, smax


def _validate_break_constraints(
    n: int,
    required_breaks: Sequence[int],
    forbidden_breaks: Sequence[int],
) -> tuple[List[int], frozenset[int]]:
    """校验必翻/禁翻序号：元素须为 int，范围 1..n-1，各自不重复、互不交叉。

    :raises PlanningError: 任一结构约束不满足。
    """
    required: List[int] = []
    seen: set[int] = set()
    for b in required_breaks:
        if not isinstance(b, int) or isinstance(b, bool):
            raise PlanningError(f"required_breaks 元素必须为整数序号（收到 {b!r}）。")
        if b < 1 or b > n - 1:
            raise PlanningError(
                f"required_breaks 含越界序号 {b}：合法范围为 1..{n - 1}"
                "（0 是首页起点、末尾序号代表文档结束，均不能作为断页位置）。"
            )
        if b in seen:
            raise PlanningError(f"required_breaks 含重复序号 {b}。")
        seen.add(b)
        required.append(b)
    required.sort()

    forbidden: set[int] = set()
    for b in forbidden_breaks:
        if not isinstance(b, int) or isinstance(b, bool):
            raise PlanningError(f"forbidden_breaks 元素必须为整数序号（收到 {b!r}）。")
        if b < 1 or b > n - 1:
            raise PlanningError(
                f"forbidden_breaks 含越界序号 {b}：合法范围为 1..{n - 1}"
                "（0 是首页起点、末尾序号代表文档结束，均不能作为断页位置）。"
            )
        if b in forbidden:
            raise PlanningError(f"forbidden_breaks 含重复序号 {b}。")
        forbidden.add(b)

    clash = sorted(seen & forbidden)
    if clash:
        raise PlanningError(
            f"序号 {clash[0]} 同时出现在 required_breaks 与 forbidden_breaks 中，"
            "必翻与禁翻约束不能交叉。"
        )
    return required, frozenset(forbidden)


def plan_pages(
    heights: Sequence[float],
    silences: Sequence[float],
    page_height: float,
    min_turn_time: float,
    required_breaks: Sequence[int] = (),
    forbidden_breaks: Sequence[int] = (),
) -> Plan:
    """计算最优翻页方案。

    ``required_breaks`` 为必翻位置（新页起始小节的 0 基序号），必须恰好成为
    断页；``forbidden_breaks`` 为禁翻位置，必须恰好不是断页。三个优化目标
    （页数、最短翻页静默、断页序列字典序）始终在全篇所有满足约束的方案中
    统一比较，不按必翻点分段取局部最优。

    所有数值按十进制字面量解释并做精确十进制比较（见模块文档"数值表示"节）：
    略超页高/略短静默一律不合格，0.1 + 0.2 恰达页高 0.3 仍可同页。

    :raises PlanningError: 数值含 NaN/Infinity、约束序号非法、必翻点静默不足、
        单小节超高，或不存在任何合法断页方案。
    """
    n = len(heights)

    # 非有限数值无法做十进制精确比较，直接判定输入非法。
    for i, h in enumerate(heights):
        if not math.isfinite(h):
            raise PlanningError(
                f"第 {i + 1} 小节高度不是有限数值（{h!r}）：不接受 NaN 或 Infinity。"
            )
    for i, s in enumerate(silences):
        if not math.isfinite(s):
            raise PlanningError(
                f"第 {i + 1} 小节后的静默秒数不是有限数值（{s!r}）："
                "不接受 NaN 或 Infinity。"
            )
    if not math.isfinite(page_height):
        raise PlanningError(
            f"页面可用高度不是有限数值（{page_height!r}）：不接受 NaN 或 Infinity。"
        )
    if not math.isfinite(min_turn_time):
        raise PlanningError(
            f"最低翻页时间不是有限数值（{min_turn_time!r}）：不接受 NaN 或 Infinity。"
        )

    # 高度与页高、静默与最低翻页时间分别放大为同比例整数（定点十进制），
    # 后续所有比较——单节/整页容量、最低翻页静默、最短静默阈值搜索与
    # 断页重构——都是同一套十进制表示下的整数精确比较，不使用浮点容差。
    scaled_heights, height_scale = _to_scaled_ints([page_height, *heights])
    i_page_height = scaled_heights[0]
    i_heights = scaled_heights[1:]
    scaled_silences, _ = _to_scaled_ints([min_turn_time, *silences])
    i_min_turn = scaled_silences[0]
    i_silences = scaled_silences[1:]

    for i, ih in enumerate(i_heights):
        if ih > i_page_height:
            raise PlanningError(
                f"第 {i + 1} 小节高度 {heights[i]} 超过页面可用高度"
                f" {page_height}，小节不可拆开，无法排版。"
            )

    required, forbidden = _validate_break_constraints(
        n, required_breaks, forbidden_breaks
    )

    # 必翻位置本身必须是合法翻页点：其后静默不足则不存在满足约束的方案。
    quiet_required = [b for b in required if i_silences[b - 1] < i_min_turn]
    if quiet_required:
        b = quiet_required[0]
        raise PlanningError(
            f"必翻位置 {b}（第 {b} 小节之后）静默仅 {silences[b - 1]} 秒，"
            f"低于最低翻页时间 {min_turn_time} 秒，必翻点静默不足，无合法方案。"
        )

    prefix: List[int] = [0]
    for ih in i_heights:
        prefix.append(prefix[-1] + ih)

    # 断页位置 b（下一页从第 b 小节开始，0 基）：
    # b == 0 是起点占位；内部翻页看第 b-1 小节后的静默；b == n 为文档结束，
    # 不作为翻页点。禁翻位置恒不允许；必翻位置若静默达标（已前置校验）则允许。
    def baseline_allowed(b: int) -> bool:
        if b <= 0 or b >= n or b in forbidden:
            return False
        return i_silences[b - 1] >= i_min_turn

    # floor_forced[r]：<= r 的最后一个必翻点（0 表示无）。页 [l..r] 内含必翻点
    # q（l <= q <= r）时唯一合法情形是 l == q，即 l >= floor_forced[r]。
    floor_forced: List[int] = [0] * n
    cursor = 0
    last = 0
    for r in range(n):
        while cursor < len(required) and required[cursor] <= r:
            last = required[cursor]
            cursor += 1
        floor_forced[r] = last

    f = _min_pages_forward(n, prefix, i_page_height, baseline_allowed, floor_forced)
    total = f[n - 1]
    if total >= INF:
        raise PlanningError(
            "无合法翻页方案：在满足最低翻页时间、必翻/禁翻约束的位置无法完成"
            "分页（存在小节无法与相邻小节排入同一页且其后静默不足，"
            "或必翻/禁翻位置相互冲突）。"
        )

    if total == 1:
        page = Page(
            page_index=1,
            start_measure=0,
            end_measure=n - 1,
            measure_count=n,
            used_height=prefix[n] / height_scale,
            turn_silence=None,
        )
        return Plan(pages=[page], page_count=1, shortest_turn_silence=None, breaks=[])

    # 下一步起需要"严格大于 i 的第一个必翻点"，正/反各做一次 O(n) 预处理。
    # next_forced[i]：以 i 为页首的页尾 e 不得越过该必翻点（e <= next_forced[i]），
    # 必翻点因此强制成为下一页页首。
    next_forced: List[int] = [n] * (n + 1)
    cursor = len(required) - 1
    nxt = n
    for i in range(n, -1, -1):
        while cursor >= 0 and required[cursor] > i:
            nxt = required[cursor]
            cursor -= 1
        next_forced[i] = nxt

    # ---- 第二步：最大化最短翻页静默（全篇统一，不按必翻点分段） ----
    candidates = sorted(
        {i_silences[b - 1] for b in range(1, n) if baseline_allowed(b)}
    )

    def feasible(threshold: int) -> bool:
        def allowed_t(b: int) -> bool:
            if b <= 0 or b >= n or b in forbidden:
                return False
            return i_silences[b - 1] >= threshold

        ft = _min_pages_forward(
            n, prefix, i_page_height, allowed_t, floor_forced
        )
        return ft[n - 1] <= total

    lo, hi = 0, len(candidates) - 1  # candidates[lo] 必然可行（等于基线）
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(candidates[mid]):
            lo = mid
        else:
            hi = mid - 1
    best_threshold = candidates[lo]

    # ---- 第三步：阈值固定，贪心取字典序最小断页序列 ----
    def best_allowed(b: int) -> bool:
        if b <= 0 or b >= n or b in forbidden:
            return False
        return i_silences[b - 1] >= best_threshold

    smin, smax = _suffix_min_max_pages(
        n, prefix, i_page_height, best_allowed, next_forced
    )

    breaks: List[int] = []
    start = 0
    remaining = total
    while remaining > 1:
        # 本页结束后下一断页 b（页 [start..b-1]），之后后缀用 remaining-1 页排完。
        max_height = prefix[start] + i_page_height
        max_b = bisect_right(prefix, max_height, start + 1, n + 1) - 1
        # 本页不得越过 start 之后的第一个必翻点：必翻点强制作为下一页页首，
        # 故候选 b 最大到 next_forced[start]（恰为该必翻点时即在此必翻）。
        max_b = min(max_b, next_forced[start])
        chosen: Optional[int] = None
        need = remaining - 1
        for b in range(start + 1, max_b + 1):
            if best_allowed(b) and smin[b] <= need <= smax[b]:
                chosen = b
                break
        if chosen is None:  # 理论上不可达：阈值方案已被证明可行
            raise PlanningError("内部错误：重构翻页方案失败。")
        breaks.append(chosen)
        start = chosen
        remaining -= 1

    if not set(required) <= set(breaks):  # 防御：必翻点必须全部出现
        raise PlanningError("内部错误：重构翻页方案遗漏必翻位置。")

    # ---- 组装页信息（整数按比例换算回浮点输出） ----
    boundaries = [0] + breaks + [n]
    pages: List[Page] = []
    turn_values: List[float] = []
    for idx in range(len(boundaries) - 1):
        a, e = boundaries[idx], boundaries[idx + 1]
        is_last = idx == len(boundaries) - 2
        silence = None if is_last else float(silences[e - 1])
        if silence is not None:
            turn_values.append(silence)
        pages.append(
            Page(
                page_index=idx + 1,
                start_measure=a,
                end_measure=e - 1,
                measure_count=e - a,
                used_height=(prefix[e] - prefix[a]) / height_scale,
                turn_silence=silence,
            )
        )

    shortest = min(turn_values) if turn_values else None
    return Plan(
        pages=pages,
        page_count=total,
        shortest_turn_silence=shortest,
        breaks=breaks,
    )
