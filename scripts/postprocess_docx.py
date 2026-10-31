#!/usr/bin/env python3
"""论文19 docx 后处理（阶段4）

修三件事（生成器在禁改目录 `17_*\\_md2docx_tf.py`，故必须在本项目内后处理）：
  M12  JAD p10-11 要求 "Avoid vertical rules and shading within table cells"，
       而共用转换器固定用 `Table Grid`（含 insideV/left/right 单线）→ 改为只留 top/bottom/insideH。
  M14  md 的代码跨度反引号（24 处）会原样进入 docx → 全部剥掉。
  M9   补 core properties：dc:title（原为空）、dc:creator。

用法: python postprocess_docx.py <in.docx> <out.docx>
"""
import os
import shutil
import sys
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def strip_table_vertical_rules(doc):
    """去掉表格的竖线：显式写 tblBorders，只保留 top/bottom/insideH。"""
    n_tbl = 0
    for tbl in doc.tables:
        tblPr = tbl._tbl.tblPr
        # 移除已有 tblBorders
        for old in tblPr.findall(qn("w:tblBorders")):
            tblPr.remove(old)
        borders = OxmlElement("w:tblBorders")
        for tag, val in (("w:top", "single"), ("w:bottom", "single"), ("w:insideH", "single")):
            el = OxmlElement(tag)
            el.set(qn("w:val"), val)
            el.set(qn("w:sz"), "4")
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "auto")
            borders.append(el)
        for tag in ("w:left", "w:right", "w:insideV"):
            el = OxmlElement(tag)
            el.set(qn("w:val"), "none")
            el.set(qn("w:sz"), "0")
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "auto")
            borders.append(el)
        # tblBorders 必须是 tblPr 里 tblStyle 之后、tblCellMar 之前的合法位置；追加即可被 Word 接受
        tblPr.append(borders)
        # 同时把表样式的边框覆盖掉：直接换掉 tblStyle 引用为无边框样式名（保留 TableGrid 会带回 insideV）
        for st in tblPr.findall(qn("w:tblStyle")):
            tblPr.remove(st)
        n_tbl += 1
    return n_tbl


def strip_backticks(doc):
    n = 0
    for p in doc.paragraphs:
        for r in p.runs:
            if "`" in r.text:
                n += r.text.count("`")
                r.text = r.text.replace("`", "")
    for tbl in doc.tables:
        for row in tbl.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    for r in p.runs:
                        if "`" in r.text:
                            n += r.text.count("`")
                            r.text = r.text.replace("`", "")
    return n


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    doc = Document(dst)
    n_tbl = strip_table_vertical_rules(doc)
    n_bt = strip_backticks(doc)
    cp = doc.core_properties
    title = doc.paragraphs[0].text if doc.paragraphs else ""
    cp.title = title
    cp.author = "Hao Chen"
    cp.comments = "Journal of Affective Disorders submission"
    doc.save(dst)
    print(f"OK  表格去竖线 {n_tbl} 个 | 剥反引号 {n_bt} 处 | dc:title 已设为稿件标题")


if __name__ == "__main__":
    main()
