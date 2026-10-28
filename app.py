# -*- coding: utf-8 -*-
"""FissureScope - carnet pedagogique de suivi des fissures, sans diagnostic."""
from __future__ import annotations

import base64
from copy import deepcopy
from contextlib import closing
from datetime import date, datetime, timedelta
from html import escape
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import uuid
import warnings
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import plotly.graph_objects as go
from scipy import stats
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage,
    KeepTogether, Flowable, PageBreak,
)

VERSION = "2.1"
INK = "#191B20"
LIME = "#FCE5E1"  # Kept as an alias for the existing report accent.
CORAL = "#D94335"
TEAL = "#176B78"
PAPER = "#F4F5F7"
COLORS = [CORAL, TEAL, "#6753A0", "#91621F", "#374E83"]
PAGES = ["Observatoire", "Fiches fissures", "Bâtiments et repères", "Carnet de visites",
         "Comparateur photo", "Mesures et incertitudes", "Rapports", "Comprendre et tester"]
ICONS = ["grid_view", "description", "domain", "edit_note", "compare", "query_stats", "file_download", "school"]
CSV_COLS = ["date", "mesures_mm", "u_instrument_mm", "notes"]


def visual_palette(mode=None):
    """Follow the browser's native Streamlit theme, without changing shared config."""
    bridge = st.session_state.get("theme_bridge", {})
    observed = bridge.get("mode") if isinstance(bridge, dict) else getattr(bridge, "mode", None)
    dark = (mode or observed or st.context.theme.type) == "dark"
    return dict(dark=dark, paper="#141A20" if dark else "#F5F3EE",
                surface="#1D262F" if dark else "#FFFFFF", ink="#F1EFE9" if dark else "#202827",
                muted="#ABB9C3" if dark else "#5D6C70", border="#34434E" if dark else "#D9DEDA",
                soft="#25343D" if dark else "#EAF1EF", teal="#80C6D0" if dark else "#176B78",
                coral="#FF9588" if dark else CORAL)


def crack_color(c, mode=None):
    palette = visual_palette(mode)
    colors = [palette["coral"], palette["teal"], "#B5A3E3" if palette["dark"] else "#6753A0",
              "#DDB77B" if palette["dark"] else "#91621F", "#99B6E3" if palette["dark"] else "#374E83"]
    digits = re.search(r"(\d+)$", c["code"])
    index = int(digits.group(1))-1 if digits else sum(c["id"].encode("utf-8"))
    return colors[index % len(colors)]


def today():
    return datetime.now(ZoneInfo("Europe/Paris")).date()


def uid():
    return uuid.uuid4().hex


def empty_data():
    return {"schema": 1, "buildings": []}


def number(value, low, high, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} : nombre attendu.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} : nombre invalide.") from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} : valeur attendue entre {low} et {high}.")
    return value


def text(value, limit, label, required=False):
    if not isinstance(value, str):
        raise ValueError(f"{label} : texte attendu.")
    value = value.strip()
    if (required and not value) or len(value) > limit:
        raise ValueError(f"{label} : texte vide ou trop long (maximum {limit}).")
    return value


def parse_readings(value):
    """Semicolons separate readings; French decimal commas remain valid."""
    if isinstance(value, str):
        parts = re.split(r"[;\n]+", value.strip())
        if not value.strip() or any(not p.strip() for p in parts):
            raise ValueError("Saisis au moins une mesure ; sépare les valeurs par un point-virgule.")
        parts = [p.strip().replace(",", ".") for p in parts]
    elif isinstance(value, list):
        parts = value
    else:
        raise ValueError("Les mesures doivent former une liste.")
    if not 1 <= len(parts) <= 100:
        raise ValueError("De 1 à 100 mesures sont autorisées par visite.")
    return [number(p, 0, 100, "Largeur en mm") for p in parts]


def normalize_photo(raw):
    if len(raw) > 5_000_000:
        raise ValueError("Photo trop volumineuse : maximum 5 Mo.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as im:
                if im.width * im.height > 20_000_000 or im.format not in ("JPEG", "PNG"):
                    raise ValueError("Choisis une image JPEG/PNG de moins de 20 mégapixels.")
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((1400, 1400))
                out = BytesIO()
                im.save(out, "JPEG", quality=86)
                return base64.b64encode(out.getvalue()).decode("ascii")
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError("Image illisible ou trop grande.") from None


def check_photo(value):
    if not isinstance(value, str) or len(value) > 2_000_000:
        raise ValueError("Photo invalide ou trop volumineuse.")
    if not value:
        return ""
    try:
        raw = base64.b64decode(value, validate=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as im:
                if im.format not in ("JPEG", "PNG") or im.width * im.height > 20_000_000:
                    raise ValueError("Photo JPEG/PNG attendue, maximum 20 mégapixels.")
                im.verify()
    except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError("Photo de sauvegarde invalide.") from None
    return value


def validate(data):
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ValueError("Format FissureScope version 1 attendu.")
    buildings = data.get("buildings")
    if not isinstance(buildings, list) or len(buildings) > 20:
        raise ValueError("Maximum 20 bâtiments par registre.")
    out, ids, names, photos = empty_data(), set(), set(), set()
    nc = nv = 0

    def identifier(v):
        v = text(v, 64, "Identifiant", True)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", v) or v in ids:
            raise ValueError("Identifiant invalide ou dupliqué.")
        ids.add(v)
        return v

    def photo(v):
        if not isinstance(v, str):
            raise ValueError("Photo invalide.")
        if v not in photos:
            check_photo(v)
            photos.add(v)
        return v

    for b in buildings:
        if not isinstance(b, dict):
            raise ValueError("Fiche bâtiment invalide.")
        name = text(b.get("name"), 100, "Bâtiment", True)
        if name.casefold() in names:
            raise ValueError("Deux bâtiments ne peuvent pas porter le même nom.")
        names.add(name.casefold())
        bb = dict(id=identifier(b.get("id")), name=name,
                  address=text(b.get("address", ""), 200, "Emplacement"),
                  photo=photo(b.get("photo", "")), cracks=[])
        cracks = b.get("cracks")
        if not isinstance(cracks, list):
            raise ValueError("Liste des fissures invalide.")
        codes = set()
        for c in cracks:
            if not isinstance(c, dict):
                raise ValueError("Fiche fissure invalide.")
            nc += 1
            code = text(c.get("code"), 30, "Repère", True)
            if code.casefold() in codes:
                raise ValueError("Repère déjà utilisé dans ce bâtiment.")
            codes.add(code.casefold())
            cc = dict(id=identifier(c.get("id")), code=code,
                      location=text(c.get("location", ""), 200, "Localisation"),
                      x=number(c.get("x"), 0, 100, "Position X"),
                      y=number(c.get("y"), 0, 100, "Position Y"), visits=[])
            next_visit = c.get("next_visit", "")
            if next_visit:
                try:
                    planned = date.fromisoformat(next_visit)
                except (TypeError, ValueError):
                    raise ValueError("Prochaine visite : date au format AAAA-MM-JJ attendue.") from None
                if not 1900 <= planned.year <= 2100:
                    raise ValueError("Prochaine visite : année attendue entre 1900 et 2100.")
                next_visit = planned.isoformat()
            elif not isinstance(next_visit, str):
                raise ValueError("Prochaine visite : date ou texte vide attendu.")
            cc["next_visit"] = next_visit
            visits = c.get("visits")
            if not isinstance(visits, list):
                raise ValueError("Liste des visites invalide.")
            days = set()
            for v in visits:
                if not isinstance(v, dict):
                    raise ValueError("Visite invalide.")
                nv += 1
                try:
                    day = date.fromisoformat(v.get("date", ""))
                except (TypeError, ValueError):
                    raise ValueError("Date attendue au format AAAA-MM-JJ.") from None
                if day > today() or day.year < 1900 or day in days:
                    raise ValueError("Date future, trop ancienne ou déjà utilisée pour ce repère.")
                days.add(day)
                cc["visits"].append(dict(id=identifier(v.get("id")), date=day.isoformat(),
                    readings=parse_readings(v.get("readings")),
                    u=number(v.get("u"), 0, 10, "Incertitude-type instrumentale"),
                    notes=text(v.get("notes", ""), 2000, "Notes"),
                    photo=photo(v.get("photo", ""))))
            cc["visits"].sort(key=lambda v: v["date"])
            bb["cracks"].append(cc)
        out["buildings"].append(bb)
    if nc > 100 or nv > 2000:
        raise ValueError("Maximum 100 repères et 2 000 visites par registre.")
    if len(json.dumps(out, ensure_ascii=False).encode()) > 25_000_000:
        raise ValueError("Registre trop volumineux : maximum 25 Mo. Exporte une sauvegarde séparée.")
    return out


def database_path():
    root = os.environ.get("FISSURESCOPE_DATA_DIR")
    if not root:
        root = str(Path.home() / "Documents" / "Codex" / "data" / "fissurescope")
    return Path(root).expanduser() / "fissurescope.sqlite3"


def connect():
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=15)
    con.execute("CREATE TABLE IF NOT EXISTS registry (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL, payload TEXT NOT NULL)")
    con.execute("INSERT OR IGNORE INTO registry VALUES (1,0,?)", (json.dumps(empty_data()),))
    con.commit()
    return con


def load_database():
    with closing(connect()) as con:
        rev, raw = con.execute("SELECT revision,payload FROM registry WHERE id=1").fetchone()
    return validate(json.loads(raw)), rev


def save_database(data, revision):
    data = validate(data)
    with closing(connect()) as con, con:
        cur = con.execute("UPDATE registry SET payload=?,revision=revision+1 WHERE id=1 AND revision=?",
                          (json.dumps(data, ensure_ascii=False), revision))
        if cur.rowcount != 1:
            raise ValueError("Une autre fenêtre a modifié le registre. Recharge puis recommence.")


def visit_stats(v):
    a = np.array(v["readings"], dtype=float)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else None
    se = sd / math.sqrt(len(a)) if sd is not None else 0.
    u = math.hypot(v["u"], se)
    return dict(mean=mean, n=len(a), sd=sd, se=se, u=u,
                low=mean-1.96*u, high=mean+1.96*u)


def evolution(c):
    vs = sorted(c["visits"], key=lambda v: v["date"])
    if len(vs) < 2:
        return None
    days = (date.fromisoformat(vs[-1]["date"]) - date.fromisoformat(vs[0]["date"])).days
    delta = visit_stats(vs[-1])["mean"] - visit_stats(vs[0])["mean"]
    return {"delta": delta, "days": days, "rate": delta / days}


def followup(c, reference=None):
    reference = reference or today()
    vs = sorted(c["visits"], key=lambda v: v["date"])
    last = vs[-1]["date"] if vs else ""
    due = c.get("next_visit", "")
    pending = bool(due and (not last or due > last))
    if pending and date.fromisoformat(due) < reference:
        label, tone = "Visite en retard", "late"
    elif not vs:
        label, tone = "Relevé incomplet", "missing"
    elif pending:
        label, tone = "Visite prévue", "planned"
    else:
        label, tone = "Suivi à jour", "current"
    return dict(label=label, tone=tone, due=due if pending else "", last=last)


def building_followup(b, reference=None):
    states = [followup(c, reference) for c in b["cracks"]]
    planned = [s["due"] for s in states if s["due"]]
    seen = [s["last"] for s in states if s["last"]]
    return dict(next=min(planned, default=""), last=max(seen, default=""),
                late=sum(s["tone"] == "late" for s in states),
                missing=sum(s["tone"] == "missing" for s in states))


def schedule_crack(c, day):
    new = deepcopy(c)
    new["next_visit"] = day.isoformat() if isinstance(day, date) else day
    return new


def build_inspection(data, draft):
    """Construct the whole inspection atomically; validate before any persistence."""
    new = deepcopy(data)
    bid, cid = draft["building_id"], draft["crack_id"]
    if draft.get("new_building"):
        b = dict(id=bid, name=draft["building_name"], address=draft["address"], photo="", cracks=[])
        new["buildings"].append(b)
    else:
        b = next((b for b in new["buildings"] if b["id"] == bid), None)
        if b is None:
            raise ValueError("Ce bâtiment n’existe plus. Recommence l’inspection.")
    b["photo"] = draft.get("reference_photo", b["photo"])
    if draft.get("new_crack"):
        c = dict(id=cid, code=draft["code"], location=draft["location"],
                 x=draft["x"], y=draft["y"], visits=[], next_visit="")
        b["cracks"].append(c)
    else:
        c = next((c for c in b["cracks"] if c["id"] == cid), None)
        if c is None:
            raise ValueError("Ce repère n’existe plus. Recommence l’inspection.")
    c["visits"].append(deepcopy(draft["visit"]))
    c["next_visit"] = draft.get("next_visit", c.get("next_visit", ""))
    return validate(new)


def uncertainty(a, b, threshold=0., draws=20000, seed=42):
    if a["date"] >= b["date"]:
        raise ValueError("Choisis une visite finale postérieure à la visite initiale.")
    threshold = number(threshold, 0, 10, "Seuil de variation")
    sa, sb = visit_stats(a), visit_stats(b)
    delta = sb["mean"] - sa["mean"]
    u = math.hypot(sa["u"], sb["u"])
    p = float(stats.norm.sf(threshold, loc=delta, scale=u)) if u else float(delta > threshold)
    rng = np.random.default_rng(seed)
    samples = rng.normal(delta, u, draws) if u else np.full(draws, delta)
    return dict(delta=delta, u=u, low=delta-1.96*u, high=delta+1.96*u,
                p=p, simulated_p=float(np.mean(samples > threshold)), samples=samples)


def visits_frame(b):
    rows = []
    for c in b["cracks"]:
        for v in sorted(c["visits"], key=lambda v: v["date"]):
            s = visit_stats(v)
            rows.append(dict(batiment=b["name"], repere=c["code"], date=v["date"],
                             mesures_mm=";".join(f"{x:g}" for x in v["readings"]),
                             u_instrument_mm=v["u"], moyenne_mm=s["mean"], n=s["n"],
                             ecart_type_mm=s["sd"], u_combinee_mm=s["u"], notes=v["notes"]))
    return pd.DataFrame(rows, columns=["batiment", "repere", *CSV_COLS[:3], "moyenne_mm", "n",
                                       "ecart_type_mm", "u_combinee_mm", "notes"])


def safe_frame(df):
    def safe(v):
        if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + v
        return v
    return df.map(safe)


def csv_bytes(df):
    return safe_frame(df).to_csv(index=False, sep=",").encode("utf-8-sig")


def import_visits(raw, suffix, c):
    if len(raw) > 2_000_000:
        raise ValueError("Maximum 2 Mo pour cet import.")
    if suffix.lower() == "xlsx":
        try:
            with zipfile.ZipFile(BytesIO(raw)) as z:
                if sum(i.file_size for i in z.infolist()) > 15_000_000:
                    raise ValueError("Classeur décompressé trop volumineux.")
            df = pd.read_excel(BytesIO(raw), dtype=str, keep_default_na=False)
        except (zipfile.BadZipFile, OSError):
            raise ValueError("Classeur Excel invalide.") from None
    else:
        try:
            df = pd.read_csv(BytesIO(raw), sep=None, engine="python", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            raise ValueError("CSV UTF-8 attendu, séparateur virgule ou point-virgule.") from None
    if not set(CSV_COLS).issubset(df.columns) or not 1 <= len(df) <= 500:
        raise ValueError("Colonnes requises : " + ", ".join(CSV_COLS) + " ; de 1 à 500 lignes.")
    new = deepcopy(c)
    for _, row in df.iterrows():
        day = row["date"]
        if suffix.lower() == "xlsx":
            try:
                parsed = datetime.fromisoformat(day)
                if parsed.hour or parsed.minute or parsed.second or parsed.microsecond or parsed.tzinfo:
                    raise ValueError("La date Excel ne doit pas inclure une heure.")
                day = parsed.date().isoformat()
            except (TypeError, ValueError):
                raise ValueError("Date Excel invalide : date civile sans heure attendue.") from None
        new["visits"].append(dict(id=uid(), date=day, readings=parse_readings(row["mesures_mm"]),
                                 u=number(row["u_instrument_mm"].replace(",", "."), 0, 10, "Incertitude"),
                                 notes=row["notes"], photo=""))
    # The full-registry validator enforces unique dates and record limits before commit.
    return new


@st.cache_data(max_entries=6, show_spinner=False)
def wall_image(stage=0):
    """A deliberately synthetic inspection plate, not a photograph or real survey."""
    rng = np.random.default_rng(72)
    w, h = 1200, 770
    noise = rng.normal(0, 2.5, (h, w, 1))
    grid = np.ones((h, w, 3)) * [215, 218, 222] + noise
    im = Image.fromarray(np.uint8(np.clip(grid, 0, 255)))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 195, h), fill="#9FAEB9")
    d.rectangle((200, 0, 208, h), fill="#BEC5BC")
    d.rectangle((800, 0, 810, h), fill="#BEC5BC")
    d.rectangle((0, 580, w, h), fill="#BAC3B9")
    for y in range(38, 580, 115):
        d.line((210, y, w, y), fill="#BBC3BA", width=2)
    d.rectangle((290, 170, 515, 400), fill="#82948D")
    d.rectangle((303, 183, 502, 388), fill="#394C49")
    d.line((402, 183, 402, 388), fill="#A9B9AC", width=9)
    d.line((303, 285, 502, 285), fill="#A9B9AC", width=9)
    path = [(610, 60), (600, 140), (626, 200), (610, 270), (653, 335),
            (630, 401), (662, 470), (644, 540), (675, 599)]
    d.line(path, fill="#48524B", width=3 + stage * 2)
    d.line([(626, 200), (690, 237), (714, 285), (768, 305)], fill="#657065", width=2 + stage)
    d.line([(880, 360), (852, 395), (899, 428), (880, 462)], fill="#536258", width=3)
    d.rectangle((45, 45, 146, 140), outline="#F7F9F4", width=3)
    d.line((60, 120, 126, 60), fill=CORAL, width=5)
    font_path = Path("C:/Windows/Fonts/consola.ttf")
    f = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default(22)
    d.text((34, 650), "FICTIF / SCHEMA", fill="#24342B", font=f)
    d.text((850, 700), "VUE DU MUR", fill="#24342B", font=f)
    out = BytesIO()
    im.save(out, "JPEG", quality=90)
    return base64.b64encode(out.getvalue()).decode("ascii")


def demo_data():
    anchor = today() - timedelta(days=30)
    b = dict(id=uid(), name="Pavillon Atelier", address="Site pédagogique fictif", photo=wall_image(), cracks=[])
    for i, (code, loc, x, y, values) in enumerate([
        ("F-01", "Façade est - au droit de la fenêtre", 52, 43, [.30, .45, .60]),
        ("F-02", "Façade est - partie droite", 73, 54, [.20, .21, .20]),
    ]):
        c = dict(id=uid(), code=code, location=loc, x=x, y=y, visits=[])
        c["next_visit"] = (today()+timedelta(days=7)).isoformat() if i == 0 else ""
        for j, val in enumerate(values):
            c["visits"].append(dict(id=uid(), date=(anchor+timedelta(days=j*15)).isoformat(),
                readings=[round(val-.01, 3), val, round(val+.01, 3)], u=.02,
                notes="Mesures et illustration fictives pour la démonstration.", photo=wall_image(j if i == 0 else 0)))
        b["cracks"].append(c)
    return validate({"schema": 1, "buildings": [b]})


PHOTO_JS = """
export default function(component) {
  const {parentElement, data, setTriggerValue} = component;
  const root = parentElement.querySelector('.plate');
  root.replaceChildren();
  root.style.aspectRatio = `${data.w}/${data.h}`;
  root.style.width=data.modal?`min(100%, calc(52vh * ${data.w/data.h}))`:'100%';
  root.style.marginInline=data.modal?'auto':'0';
  const scenes=[];
  function layer(photo, label, after=false) {
    const wrap=document.createElement('div');wrap.className=after?'layer after':'layer';
    const scene=document.createElement('div');scene.className='scene';
    const img=document.createElement('img');img.src='data:image/jpeg;base64,'+photo;img.alt=label;
    scene.appendChild(img);wrap.appendChild(scene);root.appendChild(wrap);scenes.push(scene);
    return wrap;
  }
  layer(data.photo, data.compare?'Vue initiale':'Vue de référence du bâtiment');
  const zoom=parentElement.querySelector('.zoom');
  const range=parentElement.querySelector('.reveal');
  const reset=parentElement.querySelector('.reset');
  const expand=parentElement.querySelector('.expand');
  expand.hidden=!data.enlarge;
  expand.onclick=()=>setTriggerValue('enlarge',true);
  const zoomText=parentElement.querySelector('.zoom-value');
  let panX=50,panY=50;
  zoom.value=1;
  function applyZoom(){
    const z=Number(zoom.value);
    for(const scene of scenes)scene.style.transform=`translate(${-panX*(z-1)}%,${-panY*(z-1)}%) scale(${z})`;
    zoomText.textContent=z.toFixed(1)+'×';root.style.cursor=z>1?'grab':(data.edit?'crosshair':'default');
  }
  zoom.oninput=applyZoom;
  reset.onclick=()=>{zoom.value=1;panX=panY=50;applyZoom();};
  let drag=null,moved=false;
  root.onpointerdown=e=>{
    moved=false;
    if(Number(zoom.value)<=1||e.target.closest('.pin'))return;
    drag={x:e.clientX,y:e.clientY,px:panX,py:panY};root.setPointerCapture(e.pointerId);
  };
  root.onpointermove=e=>{
    if(!drag)return;
    const r=root.getBoundingClientRect(),z=Number(zoom.value);
    if(Math.abs(e.clientX-drag.x)+Math.abs(e.clientY-drag.y)>4)moved=true;
    panX=Math.max(0,Math.min(100,drag.px-100*(e.clientX-drag.x)/(r.width*(z-1))));
    panY=Math.max(0,Math.min(100,drag.py-100*(e.clientY-drag.y)/(r.height*(z-1))));applyZoom();
  };
  root.onpointerup=()=>{drag=null;};root.onpointercancel=()=>{drag=null;};
  if (data.compare) {
    const top=layer(data.compare,'Vue finale',true);
    const line = document.createElement('div'); line.className='split'; root.appendChild(line);
    parentElement.querySelector('.compare-controls').hidden=false;
    const apply=()=>{top.style.clipPath=`inset(0 0 0 ${range.value}%)`; line.style.left=range.value+'%';};
    range.oninput=apply; apply();
  } else {
    parentElement.querySelector('.compare-controls').hidden=true;
    for(const mark of data.marks){
      const dot=document.createElement('span');dot.className='spot';
      dot.style.left=mark.x+'%';dot.style.top=mark.y+'%';scenes[0].appendChild(dot);
      const pin=document.createElement(data.edit||!data.selectable?'span':'button');
      pin.className='pin'+(mark.id===data.selected?' selected':'');pin.textContent=mark.code;
      pin.style.left=`clamp(60px, ${mark.x}%, calc(100% - 60px))`;
      pin.style.top=`clamp(25px, ${mark.y}%, calc(100% - 25px))`;
      pin.title=mark.location;
      if(!data.edit&&data.selectable){pin.type='button';pin.setAttribute('aria-label','Ouvrir le repère '+mark.code);
        pin.onclick=e=>{e.stopPropagation();setTriggerValue('chosen',mark.id);};}
      scenes[0].appendChild(pin);
    }
    if(data.point){
      const target=document.createElement('span');target.className='target';
      target.style.left=data.point.x+'%';target.style.top=data.point.y+'%';scenes[0].appendChild(target);
    }
    root.onclick=data.edit?(e)=>{
      if(moved)return;
      const r=root.getBoundingClientRect(),z=Number(zoom.value);
      const x=100*(e.clientX-r.left)/r.width/z+panX*(1-1/z);
      const y=100*(e.clientY-r.top)/r.height/z+panY*(1-1/z);
      setTriggerValue('point',{x:Math.round(Math.max(0,Math.min(100,x))*10)/10,
                               y:Math.round(Math.max(0,Math.min(100,y))*10)/10});
    }:null;
  }
  applyZoom();parentElement.querySelector('.legend').textContent=data.caption;
}
"""
PHOTO_CSS = """
.plate {position:relative;width:100%;overflow:hidden;border-radius:10px;background:#28383D;touch-action:pan-y;}
.layer,.scene {position:absolute;inset:0;}.scene {transform-origin:0 0;}
img {display:block;width:100%;height:100%;object-fit:contain;pointer-events:none;}
.pin {position:absolute;transform:translate(-50%,-50%);background:#191B20;color:white;
  border:2px solid white;padding:6px 9px;border-radius:3px;font:700 12px 'Segoe UI',sans-serif;
  max-width:115px;overflow-wrap:anywhere;cursor:pointer;}
.pin.selected,.pin:hover {background:#D94335;}
.pin:focus-visible {outline:3px solid #176B78;outline-offset:3px;}
.spot {position:absolute;width:9px;height:9px;background:#D94335;border:2px solid white;
  transform:translate(-50%,-50%);border-radius:50%;pointer-events:none;}
.target {position:absolute;width:18px;height:18px;border:3px solid #E64130;
 transform:translate(-50%,-50%);border-radius:50%;box-shadow:0 0 0 3px white;pointer-events:none;}
.split {position:absolute;top:0;bottom:0;border-left:3px solid #D94335;pointer-events:none;}
.toolbar {display:flex;align-items:center;gap:8px;padding:10px 0;border-bottom:1px solid var(--st-border-color,#D7DBE2);flex-wrap:wrap;}
.toolbar label {display:flex;align-items:center;gap:8px;font:12px 'Segoe UI',sans-serif;color:var(--st-text-color,#4E5460);flex:1;min-width:150px;}
input {min-width:0;accent-color:#176B78;cursor:ew-resize;}
.zoom {flex:1;max-width:150px;}.zoom-value {font:12px Consolas,monospace;min-width:34px;}
.reset,.expand {border:1px solid var(--st-border-color,#B9BEC9);background:var(--st-secondary-background-color,#FFF);color:var(--st-text-color,#191B20);padding:8px 10px;border-radius:6px;cursor:pointer;min-height:36px;}
.reset:hover,.expand:hover {border-color:#D94335;}.expand[hidden]{display:none;}
.reset:focus-visible,.expand:focus-visible {outline:3px solid #579FAA;outline-offset:2px;}
.compare-controls {font:12px 'Segoe UI',sans-serif;color:var(--st-text-color,#4E5460);margin-top:10px;}
.compare-controls[hidden] {display:none;}.compare-controls input {display:block;width:100%;margin:10px 0;}
.legend {font:12px 'Segoe UI',sans-serif;color:var(--st-text-color,#626875);opacity:.8;margin:8px 0 0;overflow-wrap:anywhere;}
"""
PHOTO = st.components.v2.component("fissurescope_plate", html="""
<div class="plate"></div>
<div class="compare-controls"><span>Avant</span> / <span>Après</span>
<input class="reveal" type="range" min="0" max="100" value="50" aria-label="Position du comparateur avant après"></div>
<div class="toolbar"><label>Zoom <input class="zoom" type="range" min="1" max="4" step="0.1" value="1" aria-label="Zoom synchronisé"><output class="zoom-value">1.0×</output></label>
<button type="button" class="reset">Réinitialiser</button><button type="button" class="expand" aria-label="Agrandir la vue">Agrandir ↗</button></div><p class="legend"></p>
""", css=PHOTO_CSS, js=PHOTO_JS)


def show_photo(photo, marks=None, key="plate", edit=False, point=None, compare="", caption="", building_id=None, selected=None, enlarge=True, modal=False):
    if not photo:
        photo = wall_image()
        caption = "Illustration fictive - aucune photo de ce bâtiment enregistrée."
    with Image.open(BytesIO(base64.b64decode(photo))) as im:
        w, h = im.size
    result = PHOTO(data=dict(photo=photo, w=w, h=h, marks=marks or [], edit=edit,
                           point=point, compare=compare, caption=caption, selected=selected,
                           selectable=bool(building_id), enlarge=enlarge and not edit, modal=modal),
                 key=key, on_point_change=lambda: None,
                 on_chosen_change=lambda: photo_choice(key, building_id), on_enlarge_change=lambda: None,
                 width="stretch", height="content")
    if result.enlarge and enlarge and not edit:
        photo_lightbox(photo, marks or [], caption, compare, selected)
    return result


@st.dialog("Vue photographique", width="large")
def photo_lightbox(photo, marks, caption, compare="", selected=None):
    st.caption(caption)
    show_photo(photo, marks, key="large_reference", compare=compare, caption=caption,
               selected=selected, enlarge=False, modal=True)
    if st.button("Fermer la vue", icon=":material/close:"):
        st.rerun()


THEME_BRIDGE = st.components.v2.component("fissurescope_theme", html='<span class="theme-probe"></span>',
    css="""
    .theme-probe {color:var(--st-background-color);position:absolute;width:0;height:0;overflow:hidden;}
    """, js="""
export default function({parentElement,data,setStateValue}) {
  // Native theme switches do not always rerun Python. Observe the public theme token.
  let last=data.mode;
  const probe=parentElement.querySelector('.theme-probe');
  function check(){
    const rgb=getComputedStyle(probe).color.match(/[\\d.]+/g);
    if(!rgb||rgb.length<3)return;
    const brightness=.2126*Number(rgb[0])+.7152*Number(rgb[1])+.0722*Number(rgb[2]);
    const mode=brightness<128?'dark':'light';
    if(mode!==last){last=mode;setStateValue('mode',mode);}
  }
  check();const timer=setInterval(check,400);
  return ()=>clearInterval(timer);
}
""")


def photo_choice(key, building_id):
    result = st.session_state.get(key)
    cid = getattr(result, "chosen", None) if result is not None else None
    if not cid and isinstance(result, dict):
        cid = result.get("chosen")
    if building_id and cid:
        open_record(building_id, cid, "Fiches fissures")


CSS = """
<style>
[data-testid="stAppViewContainer"] {background:#F4F5F7;color:#191B20;}
[data-testid="stHeader"] {background:rgba(244,245,247,.98);}
.block-container {max-width:1510px;padding-top:4.4rem;padding-bottom:2rem;}
h1,h2,h3,h4,p,span,button,label {letter-spacing:0!important;}
h1,h2,h3 {font-family:'Segoe UI',sans-serif;overflow-wrap:anywhere;}
h1 {font-size:28px!important;padding-top:0!important;} h2 {font-size:22px!important;} h3 {font-size:18px!important;}
[data-testid="stSidebar"] {background:#191B20;}
[data-testid="stCaptionContainer"] {color:#626875;opacity:1;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#BFC5CF;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] label,[data-testid="stSidebar"] [data-testid="stIconMaterial"] {color:#F4F5F7;}
[data-testid="stSidebar"] button {border:1px solid transparent;background:transparent;color:#E2E5EA;justify-content:flex-start;transition:background .16s ease,border-color .16s ease;}
[data-testid="stSidebar"] button:hover {border-color:#707986;background:#282C34;}
[data-testid="stSidebar"] button[kind="primary"] {background:#D94335!important;color:#FFF!important;border-color:#D94335!important;}
[data-testid="stSidebar"] button[kind="primary"] p,[data-testid="stSidebar"] button[kind="primary"] span {color:#FFF!important;}
[data-testid="stSidebar"] [data-baseweb="select"]>div {background:#F4F5F7;color:#191B20;}
[data-testid="stSidebar"] [data-baseweb="select"] span {color:#191B20;}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button {border-radius:4px;min-height:42px;white-space:normal;transition:background .16s ease,border-color .16s ease;}
[data-testid="stButton"] button p,[data-testid="stFormSubmitButton"] button p {white-space:normal;overflow-wrap:anywhere;}
button[kind="primary"],button[kind="primaryFormSubmit"] {background:#D94335!important;color:#FFF!important;border-color:#D94335!important;}
button[kind="primary"]:hover,button[kind="primaryFormSubmit"]:hover {background:#B83328!important;}
button:focus-visible,input:focus-visible {outline:3px solid #176B78!important;outline-offset:3px;}
[data-testid="stForm"] {border-color:#D4D8E0;border-radius:4px;background:#FFF;}
.mast {display:flex;align-items:center;justify-content:space-between;gap:14px;border-bottom:1px solid #C9CED7;padding:0 0 17px;margin-bottom:4px;flex-wrap:wrap;}
.brand {font:800 30px 'Segoe UI',sans-serif;line-height:1.15;color:#191B20;}
.brand em {font-style:normal;color:#D94335;}
.stamp {font:600 11px Consolas,monospace;border-left:3px solid #D94335;background:#FFF;color:#414956;padding:8px 11px;}
.eyebrow {font:600 11px Consolas,monospace;color:#626875;margin-bottom:4px;text-transform:uppercase;}
.subtitle {font-size:12px;color:#626875;margin-top:5px;}
.strip {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin:0 0 14px;border-top:1px solid #D4D8E0;border-bottom:1px solid #D4D8E0;}
.stat {min-width:0;padding:10px 15px;border-right:1px solid #D4D8E0;}
.stat:first-child{padding-left:0;}.stat:last-child {border-right:0;}
.stat b {display:block;font:700 24px Consolas,monospace;color:#191B20;margin:2px 0;overflow-wrap:anywhere;}
.stat small {font-size:11px;color:#626875;}.stat:nth-child(4) b{color:#176B78;}
.strip.compact {grid-template-columns:repeat(2,minmax(0,1fr));}
.strip.compact .stat:nth-child(2){border-right:0;}.strip.compact .stat:nth-child(-n+2){border-bottom:1px solid #D4D8E0;}
.strip.compact .stat{padding:10px;}.strip.compact .stat b{font-size:22px;}
.site {display:flex;justify-content:space-between;gap:15px;border-bottom:1px solid #D4D8E0;padding:12px 0;align-items:center;}
.site h3{margin:0;font-size:18px!important;}.site p{margin:4px 0 0;color:#626875;font-size:12px;overflow-wrap:anywhere;}
.site .index {font:700 25px Consolas,monospace;color:#176B78;min-width:42px;}
.note {padding:12px 14px;border-left:3px solid #176B78;background:#E6EFF2;margin:10px 0;font-size:13px;color:#283440;overflow-wrap:anywhere;}
.timeline {border-left:1px solid #BAC2CF;margin:0 0 0 6px;padding-left:18px;}
.entry {position:relative;padding:0 0 20px;}
.entry:before {content:'';position:absolute;left:-24px;top:5px;width:9px;height:9px;background:#D94335;border:2px solid #F4F5F7;border-radius:50%;}
.entry strong {display:block;font:700 22px Consolas,monospace;color:#191B20;}
.entry time {font-size:11px;color:#626875;}.entry p {font-size:12px;color:#626875;margin:3px 0;overflow-wrap:anywhere;}
.reading {display:flex;justify-content:space-between;align-items:center;gap:10px;border-bottom:1px solid #D4D8E0;padding:12px 0;}
.reading b {font:700 25px Consolas,monospace;color:#191B20;white-space:nowrap;}
.reading small {display:block;font-size:12px;color:#626875;margin-top:5px;overflow-wrap:anywhere;}
.reading strong {font-size:14px;}
.follow {display:inline-block;font:600 11px 'Segoe UI',sans-serif;padding:5px 8px;border-radius:3px;background:#E6EDF0;color:#23535F;overflow-wrap:anywhere;}
.follow.late{background:#FCE5E1;color:#9F2A20;}.follow.missing{background:#FFF0D2;color:#775218;}
.follow.current{background:#E1EEEB;color:#275D54;}.follow.planned{background:#E3EDF4;color:#29536B;}
.steps{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border-bottom:1px solid #CCD2DB;margin:6px 0 18px;}
.step{padding:9px 6px;color:#626875;font-size:12px;border-bottom:3px solid transparent;overflow-wrap:anywhere;}
.step.active{border-color:#D94335;color:#191B20;font-weight:700;}.step.done{color:#176B78;}
.photo-ribbon{display:flex;gap:12px;overflow-x:auto;padding:3px 0 10px;}
.photo-ribbon figure{margin:0;flex:0 0 132px;}.photo-ribbon img{width:132px;height:88px;object-fit:contain;background:#E1E4E9;border-radius:3px;}
.photo-ribbon figcaption{font-size:11px;color:#626875;padding-top:5px;}
.footer {border-top:1px solid #D4D8E0;margin-top:25px;padding-top:12px;font-size:11px;color:#626875;}
@media(prefers-reduced-motion:reduce){*{transition:none!important;}}
@media(max-width:900px){.block-container{padding-left:1.1rem;padding-right:1.1rem;} .stat{padding:10px;}.stat b{font-size:22px;}}
@media(max-width:640px){.block-container{padding-top:4.4rem;padding-left:1rem;padding-right:1rem;}.brand{font-size:26px;}
.strip{grid-template-columns:repeat(2,minmax(0,1fr));}.stat:nth-child(2){border-right:0;}.stat:nth-child(-n+2){border-bottom:1px solid #C9D0C5;}
.stat:first-child{padding-left:10px;}.stat b{font-size:22px;}.mast{gap:10px;} .stamp{font-size:10px;} h1{font-size:24px!important;}
.steps{grid-template-columns:repeat(5,minmax(0,1fr));}.step{font-size:10px;padding:9px 3px;}}
</style>
"""


ATELIER_CSS = """
<style>
[data-testid="stAppViewContainer"], [data-testid="stMain"] {background:var(--fs-paper);color:var(--fs-ink);}
[data-testid="stHeader"] {background:var(--fs-paper);}
.block-container {max-width:1580px;padding-top:4.3rem;}
.st-key-theme_bridge {display:none;}
h1,h2,h3,h4 {color:var(--fs-ink)!important;font-family:'Segoe UI',Arial,sans-serif;}
h1 {font-size:34px!important;font-weight:750!important;line-height:1.18;}
h2 {font-size:24px!important;}h3 {font-size:20px!important;}
[data-testid="stMain"] [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {color:var(--fs-ink);}
[data-testid="stCaptionContainer"] {color:var(--fs-muted);opacity:1;}
[data-testid="stForm"] {background:var(--fs-surface);border:1px solid var(--fs-border);border-radius:12px;}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button {border-radius:8px;min-height:44px;}
[data-testid="stSidebar"] {background:#19262B;border-right:1px solid #33464C;}
[data-testid="stSidebar"] button {border-radius:7px;}
[data-testid="stSidebar"] button:hover {background:#2B4148;border-color:#66818A;}
[data-testid="stSidebar"] button p,[data-testid="stSidebar"] button span,[data-testid="stSidebar"] summary,[data-testid="stSidebar"] summary p,[data-testid="stSidebar"] summary span {color:#E8F1F2!important;}
[data-testid="stSidebar"] [data-baseweb="select"]>div {background:#253D45;color:#FFF;border-color:#4B626A;}
[data-testid="stSidebar"] [data-baseweb="select"] span {color:#FFF;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#C0D1D5;}
.side-logo {display:flex;gap:11px;align-items:center;margin-bottom:16px;color:#FFF;font:750 22px 'Segoe UI',sans-serif;}
.logo-symbol {display:inline-flex;width:35px;height:40px;align-items:center;justify-content:center;background:#D94335;border-radius:7px;flex-shrink:0;}
.logo-symbol svg {width:22px;height:27px;}
.logo-symbol:before {content:'';display:block;width:19px;height:28px;background:white;clip-path:polygon(66% 0,44% 32%,73% 48%,36% 100%,22% 100%,54% 51%,25% 35%,53% 0);}
.mast {border-color:var(--fs-border);padding-bottom:16px;margin-bottom:7px;}
.brand {display:flex;align-items:center;gap:12px;font-size:32px;font-weight:800;color:var(--fs-ink);letter-spacing:-1px!important;}
.brand em {color:var(--fs-coral);}
.brand .logo-symbol {height:43px;width:38px;}.brand .wordmark {white-space:nowrap;}
.subtitle,.eyebrow {color:var(--fs-muted);}.subtitle {margin-left:50px;font-size:12px;}
.stamp {border:1px solid var(--fs-border);border-left:3px solid var(--fs-coral);border-radius:6px;color:var(--fs-muted);background:var(--fs-surface);}
.eyebrow {font-size:10px;letter-spacing:1.1px!important;margin:4px 0 0;}
.strip {background:var(--fs-surface);border:1px solid var(--fs-border);border-radius:12px;overflow:hidden;margin:0 0 22px;}
.stat {padding:12px 20px!important;border-color:var(--fs-border);border-top:3px solid var(--fs-teal);}
.stat:first-child {border-top-color:var(--fs-coral);}.stat:nth-child(3){border-top-color:#A79468;}
.stat:nth-child(4){border-top-color:#617A8A;}.stat b {font:750 30px 'Segoe UI',Arial,sans-serif;color:var(--fs-ink);font-variant-numeric:tabular-nums;line-height:1.3;margin:5px 0;}
.stat small {display:block;line-height:1.4;color:var(--fs-muted);font-size:11px;}.stat:nth-child(4) b {color:var(--fs-teal);}
.strip {margin-bottom:7px;}.stat b {line-height:1.15;}
.strip.compact .stat b {font-size:26px;}.strip.compact .stat {padding:13px!important;}
.strip.compact .stat:nth-child(-n+2) {border-bottom-color:var(--fs-border);}
.site {border:0;padding:4px 0 14px;}.site h3 {font-size:25px!important;font-weight:750;}
.site p {color:var(--fs-muted);font-size:12px;}.site .index {color:var(--fs-teal);}
.section-kicker {font:650 10px 'Segoe UI',sans-serif;letter-spacing:1.3px!important;text-transform:uppercase;color:var(--fs-muted);margin-bottom:7px;}
.st-key-inspection_focus {background:var(--fs-surface);border:1px solid var(--fs-border);padding:22px;border-radius:16px;box-shadow:0 10px 32px rgba(21,39,42,.05);}
.st-key-inspection_focus {gap:8px!important;}.st-key-inspection_focus .section-kicker{line-height:1.3;margin:0;}.st-key-inspection_focus .site{padding-bottom:7px;}
.st-key-focus_summary,.st-key-first_dossier {background:#164852;border-radius:11px;padding:22px;color:#F7F8F3;}
.st-key-focus_summary h3,.st-key-first_dossier h3,.st-key-focus_summary [data-testid="stMarkdownContainer"],.st-key-first_dossier [data-testid="stMarkdownContainer"] {color:#F7F8F3!important;}
.st-key-focus_summary [data-testid="stCaptionContainer"],.st-key-first_dossier [data-testid="stCaptionContainer"] {color:#BDD6DA!important;}
.st-key-focus_summary .reading {border-color:#42717A;padding:14px 0;}
.st-key-focus_summary .reading b {color:#FFF;font-family:'Segoe UI',sans-serif;font-size:29px;font-weight:750;}
.st-key-focus_summary .reading small {color:#BDD6DA;}.st-key-focus_summary .reading strong{font-size:13px;}
.st-key-focus_summary button[kind="secondary"],.st-key-first_dossier button[kind="secondary"] {background:#205864;border-color:#78A5AF;color:#FFF;}
.st-key-focus_summary button p,.st-key-focus_summary button span,.st-key-first_dossier button p,.st-key-first_dossier button span,.st-key-first_dossier [data-testid="stForm"] button [data-testid="stMarkdownContainer"] {color:#FFF!important;}
.st-key-first_dossier [data-testid="stForm"] {background:var(--fs-surface);color:var(--fs-ink);}
.st-key-first_dossier [data-testid="stForm"] [data-testid="stWidgetLabel"],.st-key-first_dossier [data-testid="stForm"] [data-testid="stMarkdownContainer"] {color:var(--fs-ink)!important;}
.focus-date {font:700 25px 'Segoe UI',sans-serif;color:#FFF;margin:0 0 8px;font-variant-numeric:tabular-nums;}
.reading {border-color:var(--fs-border);}.reading b {color:var(--fs-ink);}.reading small {color:var(--fs-muted);}
.note {background:var(--fs-soft);color:var(--fs-ink);border-color:var(--fs-teal);border-radius:0 8px 8px 0;}
.timeline {border-color:var(--fs-border);}.entry:before {border-color:var(--fs-paper);background:var(--fs-coral);}
.entry strong {font-family:'Segoe UI',sans-serif;color:var(--fs-ink);font-variant-numeric:tabular-nums;}
.entry time,.entry p {color:var(--fs-muted);}
.step {color:var(--fs-muted);}.step.active {color:var(--fs-ink);}.step.done {color:var(--fs-teal);}
.steps,.footer {border-color:var(--fs-border);}.footer {color:var(--fs-muted);}
.building-thumb {width:100%;height:170px;object-fit:cover;background:#CCD5D5;border-radius:10px;display:block;}
[class*="st-key-building_card_"],[class*="st-key-crack_card_"] {background:var(--fs-surface);border:1px solid var(--fs-border);border-radius:13px;padding:14px;transition:transform .18s ease,box-shadow .18s ease;}
[class*="st-key-building_card_"]:hover,[class*="st-key-crack_card_"]:hover {transform:translateY(-3px);box-shadow:0 9px 24px rgba(23,53,58,.12);}
.building-name {font:750 20px 'Segoe UI',sans-serif;color:var(--fs-ink);margin:14px 0 7px;overflow-wrap:anywhere;}
.building-meta {display:flex;justify-content:space-between;gap:8px;font-size:12px;color:var(--fs-muted);padding:10px 0;border-top:1px solid var(--fs-border);}
.inspection-label {display:flex;align-items:center;gap:9px;color:var(--fs-muted);font-size:12px;margin-bottom:9px;}
.inspection-label:before {content:'';width:8px;height:8px;border-radius:50%;background:var(--fs-coral);}
[data-testid="stDialog"] [role="dialog"] {background:var(--fs-surface);color:var(--fs-ink);border:1px solid var(--fs-border);border-radius:14px;}
[data-testid="stPlotlyChart"] {border-radius:10px;overflow:hidden;}
@media(max-width:900px){.st-key-inspection_focus{padding:16px;}.st-key-focus_summary{padding:17px;}.stat{padding:12px!important;}.stat b{font-size:26px;}}
@media(max-width:640px){h1{font-size:29px!important;}.brand{font-size:27px;gap:9px;}.brand .logo-symbol{width:31px;height:35px;}.subtitle{margin-left:40px;}
.mast{gap:12px;}.stamp{font-size:10px;}.stat{padding:12px!important;}.stat b{font-size:25px;}.stat:nth-child(-n+2){border-bottom-color:var(--fs-border);}
.st-key-inspection_focus{padding:12px;border-radius:12px;}.site{align-items:flex-start;gap:9px;flex-wrap:wrap;}.site h3{font-size:23px!important;}
.st-key-focus_summary{padding:17px;}.st-key-first_dossier{padding:16px;}.building-thumb{height:180px;}.strip.compact .stat b{font-size:23px;}}
@media(prefers-reduced-motion:reduce){[class*="st-key-building_card_"],[class*="st-key-crack_card_"]{transition:none;transform:none!important;}}
</style>
"""


def apply_visual_style():
    palette = visual_palette()
    tokens = ";".join("--fs-"+k+":"+v for k, v in palette.items() if k != "dark")
    st.html(CSS + '<style>:root {'+tokens+'}</style>' + ATELIER_CSS)


LOGO = '<span class="logo-symbol" aria-hidden="true"></span>'


def fmt(v, digits=2):
    return f"{v:.{digits}f}".replace(".", ",")


def strip(items, compact=False):
    st.html('<div class="strip'+(' compact' if compact else '')+'">'+"".join(
        f'<div class="stat"><small>{escape(str(label))}</small><b>{escape(str(value))}</b><small>{escape(str(sub))}</small></div>'
        for label, value, sub in items)+'</div>')


def note(message):
    st.html(f'<div class="note">{escape(message)}</div>')


def chart_layout(fig, height=320):
    palette = visual_palette()
    fig.update_layout(height=height, paper_bgcolor=palette["paper"], plot_bgcolor=palette["paper"],
        font=dict(family="Segoe UI", color=palette["ink"], size=12), margin=dict(l=12, r=18, t=15, b=30),
        hoverlabel=dict(bgcolor=palette["surface"], font_color=palette["ink"], bordercolor=palette["border"]),
        legend=dict(orientation="h", y=1.1), hovermode="x unified")
    fig.update_xaxes(showgrid=False, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=palette["border"], zerolinecolor=palette["border"], automargin=True)
    return fig


def evolution_chart(cracks):
    fig = go.Figure()
    for i, c in enumerate(cracks):
        vs = sorted(c["visits"], key=lambda v: v["date"])
        if not vs:
            continue
        ss = [visit_stats(v) for v in vs]
        fig.add_trace(go.Scatter(x=[v["date"] for v in vs], y=[s["mean"] for s in ss],
            customdata=[[v["id"]] for v in vs],
            name=c["code"], mode="lines+markers", line=dict(color=crack_color(c), width=2.5),
            marker=dict(size=8, color=crack_color(c)), error_y=dict(type="data", array=[1.96*s["u"] for s in ss], visible=True),
            hovertemplate="%{x}<br>%{y:.3f} mm<extra>%{fullData.name}</extra>"))
    fig.update_yaxes(title="Largeur moyenne (mm)", rangemode="tozero")
    return chart_layout(fig)


def show_evolution(cracks, key, alt):
    event = st.plotly_chart(evolution_chart(cracks), theme=None, key=key, on_select="rerun",
                           selection_mode="points", alt=alt)
    points = event["selection"]["points"]
    if points:
        custom = points[-1].get("customdata", [])
        selected = custom[0] if custom else None
        match = next(((c, v) for c in cracks for v in c["visits"] if v["id"] == selected), None)
        if match:
            c, v = match
            s = visit_stats(v)
            note(f"{c['code']} · {date_label(v['date'])} : moyenne {fmt(s['mean'], 3)} mm, "
                 f"{s['n']} répétition(s), incertitude-type {fmt(s['u'], 3)} mm. " + (v["notes"] or "Sans observation écrite."))
    st.caption("Survoler un point pour lire sa valeur ; cliquer pour afficher le détail du relevé.")


def persist(new, revision, next_page=None):
    try:
        new = validate(new)
        if st.session_state.space == "Exemple fictif":
            st.session_state.demo = new
            st.session_state.demo_revision += 1
        else:
            save_database(new, revision)
        if next_page:
            st.session_state.page = next_page
        st.session_state.flash = "Modifications enregistrées."
        st.rerun()
    except (ValueError, sqlite3.Error) as e:
        st.error(str(e))


def switch_page(page):
    st.session_state.page = page


def open_record(bid, cid=None, page="Fiches fissures"):
    st.session_state.selected_building = bid
    st.session_state.selected_crack = cid
    st.session_state.pop("building_picker", None)
    st.session_state.pop("crack_picker_"+bid, None)
    switch_page(page)


def start_inspection(bid=None, cid=None):
    st.session_state.inspection_draft = {"space": st.session_state.get("space", "Mes données"),
        "stage": 0, "building_id": bid, "crack_id": cid, "nonce": uid()}
    switch_page("Nouvelle inspection")


def status_badge(c):
    s = followup(c)
    st.html(f'<span class="follow {s["tone"]}">{escape(s["label"])}</span>')


def date_label(value):
    return date.fromisoformat(value).strftime("%d/%m/%Y") if value else "Non renseignée"


GALLERY_JS = """
export default function({parentElement,data,setTriggerValue}) {
  const strip=parentElement.querySelector('.gallery-strip');strip.replaceChildren();
  for(const entry of data.entries){
    const button=document.createElement('button');button.type='button';button.className='gallery-photo';
    button.setAttribute('aria-label','Agrandir '+entry.code+' du '+entry.date);
    const img=document.createElement('img');img.src='data:image/jpeg;base64,'+entry.photo;img.alt=entry.code+' · '+entry.date;
    const line=document.createElement('span');line.className='gallery-date';line.textContent=entry.date;
    const meta=document.createElement('span');meta.className='gallery-meta';meta.textContent=entry.code+' · '+entry.mean+' mm';
    const icon=document.createElement('span');icon.className='gallery-open';icon.textContent='↗';icon.setAttribute('aria-hidden','true');
    button.append(img,line,meta,icon);button.onclick=()=>setTriggerValue('visit',{crack:entry.crack,visit:entry.id});
    strip.appendChild(button);
  }
  const behavior=matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth';
  parentElement.querySelector('.gallery-back').onclick=()=>strip.scrollBy({left:-strip.clientWidth*.75,behavior});
  parentElement.querySelector('.gallery-next').onclick=()=>strip.scrollBy({left:strip.clientWidth*.75,behavior});
}
"""
GALLERY_CSS = """
.gallery-strip {display:flex;gap:14px;overflow:auto;scroll-snap-type:x proximity;padding:5px 2px 13px;}
.gallery-photo {position:relative;flex:0 0 170px;scroll-snap-align:start;padding:0 0 12px;border:1px solid var(--st-border-color,#D7DDDA);border-radius:10px;
  background:var(--st-secondary-background-color,#FFF);text-align:left;color:var(--st-text-color,#202827);overflow:hidden;cursor:pointer;transition:transform .16s ease;}
.gallery-photo:hover {transform:translateY(-3px);}.gallery-photo:focus-visible {outline:3px solid #298797;outline-offset:2px;}
.gallery-photo img {display:block;width:170px;height:112px;object-fit:cover;background:#CAD5D5;}
.gallery-date,.gallery-meta {display:block;font:600 12px 'Segoe UI',sans-serif;margin:10px 12px 0;}
.gallery-meta {font-weight:400;margin-top:5px;font-size:11px;opacity:.8;}
.gallery-open {position:absolute;right:8px;top:8px;background:#19262B;color:white;border-radius:5px;width:23px;height:23px;text-align:center;line-height:22px;}
.gallery-tools {display:flex;justify-content:space-between;gap:12px;align-items:center;font:12px 'Segoe UI',sans-serif;color:var(--st-text-color,#5D6C70);margin-bottom:9px;}
.gallery-tools button {width:34px;height:32px;border:1px solid var(--st-border-color,#CBD2CD);border-radius:6px;background:var(--st-secondary-background-color,#FFF);color:inherit;cursor:pointer;}
.gallery-tools button:hover{border-color:#298797;}.gallery-tools button:focus-visible{outline:3px solid #298797;}
@media(prefers-reduced-motion:reduce){.gallery-photo{transition:none;transform:none!important;}.gallery-strip{scroll-behavior:auto;}}
"""
GALLERY = st.components.v2.component("fissurescope_gallery", html="""
<div class="gallery-tools"><span>Sélectionner une photo pour ouvrir la visite</span><div><button type="button" class="gallery-back" aria-label="Photos précédentes">←</button> <button type="button" class="gallery-next" aria-label="Photos suivantes">→</button></div></div>
<div class="gallery-strip"></div>
""", css=GALLERY_CSS, js=GALLERY_JS)


@st.dialog("Photothèque d’inspection", width="large")
def visit_lightbox(b, c):
    photos = sorted([v for v in c["visits"] if v["photo"]], key=lambda v: v["date"])
    if not photos:
        st.info("Aucune photo disponible pour ce repère.")
        return
    visits = {v["id"]: v for v in photos}
    key = "lightbox_visit_"+c["id"]
    if st.session_state.get(key) not in visits:
        st.session_state[key] = photos[-1]["id"]
    vid = st.selectbox("Date de la photo", list(visits), format_func=lambda x: date_label(visits[x]["date"]), key=key)
    v = visits[vid]
    previous = sorted([vv for vv in c["visits"] if vv["date"] < v["date"]], key=lambda vv: vv["date"])
    s = visit_stats(v)
    st.subheader(b["name"]+" / "+c["code"])
    st.caption(c["location"])
    show_photo(v["photo"], key="large_visit_"+v["id"], caption="Photo de visite · "+date_label(v["date"]), enlarge=False, modal=True)
    delta = s["mean"]-visit_stats(previous[-1])["mean"] if previous else None
    strip([("Moyenne", fmt(s["mean"], 3)+" mm", "Mesure saisie"),
           ("Répétitions", s["n"], "Au même point"),
           ("Variation", ("+" if delta >= 0 else "")+fmt(delta, 3)+" mm" if delta is not None else "--", "Depuis le relevé précédent"),
           ("Incertitude-type", fmt(s["u"], 3)+" mm", "Combinée pour la visite")])
    st.write(v["notes"] or "Aucune observation écrite pour cette visite.")
    if st.button("Fermer la photo", icon=":material/close:"):
        st.rerun()


def photo_ribbon(b, c=None, limit=8):
    entries = [(v, cc) for cc in ([c] if c else b["cracks"]) for v in cc["visits"] if v["photo"]]
    entries.sort(key=lambda x: x[0]["date"], reverse=True)
    if not entries:
        st.caption("Aucune photo de visite enregistrée.")
        return
    result = GALLERY(data=dict(entries=[dict(id=v["id"], crack=cc["id"], code=cc["code"], date=date_label(v["date"]),
                         photo=v["photo"], mean=fmt(visit_stats(v)["mean"])) for v, cc in entries[:limit]]),
                     key="gallery_"+b["id"]+(c["id"] if c else "_all"), on_visit_change=lambda: None,
                     width="stretch", height="content")
    if result.visit:
        found = next((cc for cc in b["cracks"] if cc["id"] == result.visit.get("crack")), None)
        if found and any(v["id"] == result.visit.get("visit") and v["photo"] for v in found["visits"]):
            st.session_state["lightbox_visit_"+found["id"]] = result.visit["visit"]
            visit_lightbox(b, found)


def thumb(photo, alt):
    st.html(f'<img class="building-thumb" src="data:image/jpeg;base64,{photo or wall_image()}" alt="{escape(alt, quote=True)}">')


def enter_demo():
    st.session_state.demo = demo_data()
    st.session_state.demo_revision = 0
    st.session_state.space = "Exemple fictif"
    st.session_state.page = "Observatoire"
    st.session_state.selected_building = None
    st.session_state.selected_crack = None


def select_building(data):
    if not data["buildings"]:
        st.info("Crée un bâtiment dans « Bâtiments et repères », ou ouvre l’exemple fictif.")
        return None
    buildings = {b["id"]: b for b in data["buildings"]}
    value = st.session_state.get("selected_building")
    index = list(buildings).index(value) if value in buildings else 0
    bid = st.selectbox("Bâtiment", list(buildings), index=index,
                       format_func=lambda x: buildings[x]["name"], key="building_picker")
    st.session_state.selected_building = bid
    return buildings[bid]


def select_crack(b):
    if not b["cracks"]:
        st.info("Ajoute d’abord un repère dans « Bâtiments et repères ».")
        return None
    cracks = {c["id"]: c for c in b["cracks"]}
    value = st.session_state.get("selected_crack")
    index = list(cracks).index(value) if value in cracks else 0
    cid = st.selectbox("Fissure / repère", list(cracks), index=index,
                       format_func=lambda x: cracks[x]["code"]+" · "+cracks[x]["location"],
                       key=f"crack_picker_{b['id']}")
    st.session_state.selected_crack = cid
    return cracks[cid]


def dashboard(data, rev):
    buildings = data["buildings"]
    cracks = [c for b in buildings for c in b["cracks"]]
    visits = [v for c in cracks for v in c["visits"]]
    late = sum(followup(c)["tone"] == "late" for c in cracks)
    strip([("Bâtiments suivis", str(len(buildings)).zfill(2), "Dossiers d’inspection"),
           ("Repères", str(len(cracks)).zfill(2), "Fissures enregistrées"),
           ("Visites", str(len(visits)).zfill(2), "Observations datées"),
           ("Visites en retard", str(late).zfill(2), "État du suivi uniquement")])
    if not buildings:
        left, right = st.columns([1.6, 1], gap="large")
        with left:
            st.html('<div class="inspection-label">Vue d’atelier · illustration fictive</div>')
            show_photo(wall_image(), caption="Illustration fictive d’un mur - aucune donnée réelle chargée.", key="empty_plate")
        with right:
            with st.container(key="first_dossier"):
                st.subheader("Votre premier dossier")
                st.caption("Un bâtiment, ses repères et toutes les visites au même endroit.")
                with st.form("quick_building"):
                    name = st.text_input("Nom du bâtiment", placeholder="Ex. Bloc A - campus", max_chars=100)
                    address = st.text_input("Emplacement", max_chars=200)
                    if st.form_submit_button("Créer le dossier", icon=":material/add:", type="primary"):
                        new = deepcopy(data)
                        new["buildings"].append(dict(id=uid(), name=name, address=address, photo="", cracks=[]))
                        persist(new, rev)
                st.button("Explorer l’exemple fictif", icon=":material/science:", on_click=enter_demo, width="stretch")
        return
    b = next((b for b in buildings if b["id"] == st.session_state.get("selected_building")), buildings[0])
    st.session_state.selected_building = b["id"]
    overview = building_followup(b)
    with st.container(key="inspection_focus"):
        st.html('<div class="section-kicker">Dossier actif / vue de référence</div>')
        st.html(f'<div class="site"><div><h3>{escape(b["name"])}</h3><p>{escape(b["address"])} · Dernière visite : {date_label(overview["last"])}</p></div>'
                f'<span class="follow {"late" if overview["late"] else "planned"}">{overview["late"]} en retard · {len(b["cracks"])} repères</span></div>')
        left, right = st.columns([2, 1], gap="large")
        with left:
            show_photo(b["photo"], b["cracks"], key="dashboard_"+b["id"], building_id=b["id"],
                       selected=st.session_state.get("selected_crack"), caption="Vue de référence · "+b["address"])
        with right:
            with st.container(key="focus_summary"):
                st.subheader("Derniers relevés")
                for c in b["cracks"][:3]:
                    vs = sorted(c["visits"], key=lambda v: v["date"])
                    value = fmt(visit_stats(vs[-1])["mean"])+" mm" if vs else "--"
                    s = followup(c)
                    st.html(f'<div class="reading"><div><strong>{escape(c["code"])}</strong><small>{escape(s["label"])}</small></div><b>{value}</b></div>')
                if not b["cracks"]:
                    st.caption("Aucun repère enregistré.")
                st.caption("PROCHAINE VISITE")
                st.html('<div class="focus-date">'+(date_label(overview["next"]) if overview["next"] else "Non planifiée")+'</div>')
                st.button("Nouvelle inspection", icon=":material/add_a_photo:", type="primary", key="dashboard_new",
                          on_click=start_inspection, args=(b["id"], None), width="stretch")
                st.button("Fiches des fissures", icon=":material/arrow_forward:", key="dashboard_files",
                          on_click=open_record, args=(b["id"], None, "Fiches fissures"), width="stretch")
    st.subheader("Chronologie photographique")
    photo_ribbon(b)
    if len(buildings) > 1:
        st.subheader("Tous les bâtiments")
        cols = st.columns(3, gap="medium")
        for i, bb in enumerate(buildings):
            with cols[i % 3]:
                with st.container(key="building_card_"+bb["id"]):
                    thumb(bb["photo"], "Vue de référence de "+bb["name"])
                    bs = building_followup(bb)
                    st.html('<div class="building-name">'+escape(bb["name"])+ '</div>')
                    st.caption(bb["address"] or "Emplacement non renseigné")
                    st.html(f'<div class="building-meta"><span>{len(bb["cracks"])} repères</span><span>{sum(len(c["visits"]) for c in bb["cracks"])} visites</span></div>')
                    st.caption("Dernière visite : "+date_label(bs["last"])+" · Prochaine : "+date_label(bs["next"]))
                    st.button("Ouvrir le dossier", key="focus_"+bb["id"], on_click=open_record,
                              args=(bb["id"], None, "Observatoire"), width="stretch", icon=":material/arrow_outward:")
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.subheader("L’évolution en un regard")
        if visits:
            labels = []
            for b in buildings:
                for c in b["cracks"]:
                    labels.append({**c, "code": b["name"]+" / "+c["code"]})
            show_evolution(labels, "dashboard_evolution", "Largeurs moyennes des fissures par date, avec incertitudes de mesure.")
        else:
            st.caption("Aucune visite enregistrée pour le moment.")
    with right:
        st.subheader("Dernières observations")
        recent = sorted([(v, c, b) for b in buildings for c in b["cracks"] for v in c["visits"]],
                        key=lambda x: x[0]["date"], reverse=True)[:4]
        st.html('<div class="timeline">'+"".join(
            f'<div class="entry"><time>{escape(v["date"])}</time><strong>{fmt(visit_stats(v)["mean"])} mm</strong><p>{escape(b["name"])} / {escape(c["code"])}</p></div>'
            for v, c, b in recent)+'</div>')


def cracks_page(data, rev):
    b = select_building(data)
    if not b:
        return
    c = select_crack(b)
    if not c:
        return
    vs = sorted(c["visits"], key=lambda v: v["date"])
    e = evolution(c)
    st.subheader(c["code"]+" · "+c["location"])
    status_badge(c)
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        show_photo(b["photo"], [c], key="crack_plate_"+c["id"], building_id=b["id"], selected=c["id"], caption="Repère sur la vue de référence")
    with right:
        strip([("Première mesure", fmt(visit_stats(vs[0])["mean"])+" mm" if vs else "--", date_label(vs[0]["date"]) if vs else "Aucune visite"),
               ("Dernière mesure", fmt(visit_stats(vs[-1])["mean"])+" mm" if vs else "--", date_label(vs[-1]["date"]) if vs else "Aucune visite"),
               ("Variation", fmt(e["delta"])+" mm" if e else "--", "Observée, non prédite"),
               ("Visites", len(vs), "Relevés enregistrés")], compact=True)
        st.button("Inspecter cette fissure", type="primary", icon=":material/add_a_photo:", on_click=start_inspection,
                  args=(b["id"], c["id"]), key="inspect_crack", width="stretch")
        st.button("Ouvrir le carnet", icon=":material/edit_note:", on_click=open_record,
                  args=(b["id"], c["id"], "Carnet de visites"), key="crack_carnet", width="stretch")
        st.button("Comparer les photos", icon=":material/compare:", on_click=open_record,
                  args=(b["id"], c["id"], "Comparateur photo"), key="crack_compare", width="stretch")
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        st.subheader("Évolution et observations")
        if vs:
            show_evolution([c], "crack_chart_"+c["id"], "Évolution des mesures de la fissure sélectionnée.")
            for v in reversed(vs[-4:]):
                st.markdown(f"**{date_label(v['date'])} · {fmt(visit_stats(v)['mean'])} mm**")
                st.write(v["notes"] or "Sans observation écrite.")
        photo_ribbon(b, c)
    with right:
        st.subheader("Planifier le suivi")
        with st.form(f"followup_{c['id']}_{rev}_{st.session_state.space}"):
            active = st.checkbox("Planifier une prochaine visite", value=bool(c.get("next_visit")))
            due = st.date_input("Date prévue", date.fromisoformat(c["next_visit"]) if c.get("next_visit") else today()+timedelta(days=14),
                                min_value=date(1900, 1, 1), max_value=date(2100, 12, 31))
            if st.form_submit_button("Enregistrer la planification", icon=":material/event_available:"):
                persist(replace_crack(data, schedule_crack(c, due if active else "")), rev)
        st.caption("Une visite enregistrée à cette date ou après couvre l’échéance. Ces états concernent le suivi, pas la sécurité du bâtiment. Aucune notification externe n’est envoyée.")
    st.subheader("Tous les repères du bâtiment")
    cols = st.columns(3)
    for i, cc in enumerate(b["cracks"]):
        with cols[i % 3]:
            with st.container(key="crack_card_"+cc["id"]):
                thumb(b["photo"], "Vue de référence - "+cc["code"])
                st.markdown("**"+cc["code"]+"**")
                st.caption(cc["location"])
                status_badge(cc)
                series = sorted(cc["visits"], key=lambda v: v["date"])
                if series:
                    fig = go.Figure(go.Scatter(x=[v["date"] for v in series], y=[visit_stats(v)["mean"] for v in series],
                        mode="lines+markers", line=dict(color=crack_color(cc), width=2), marker=dict(size=5)))
                    p = visual_palette()
                    fig.update_layout(height=85, margin=dict(l=4,r=4,t=4,b=4), paper_bgcolor=p["surface"], plot_bgcolor=p["surface"], showlegend=False)
                    fig.update_xaxes(visible=False)
                    fig.update_yaxes(visible=False)
                    st.plotly_chart(fig, theme=None, key="spark_"+cc["id"], config={"displayModeBar":False}, alt="Petite courbe des mesures du repère "+cc["code"])
                st.button("Voir "+cc["code"], key="view_"+cc["id"], on_click=open_record,
                          args=(b["id"], cc["id"], "Fiches fissures"), width="stretch", icon=":material/arrow_forward:")


def inspection_page(data, rev):
    draft = st.session_state.get("inspection_draft")
    if not draft or draft.get("space") != st.session_state.space:
        start_inspection()
        draft = st.session_state.inspection_draft
    draft.setdefault("base_revision", rev)
    if draft["base_revision"] != rev:
        st.warning("Le registre a changé pendant cette inspection. Aucun brouillon n’a été enregistré. Recommence pour éviter d’écraser une modification.")
        st.button("Recommencer l’inspection", on_click=start_inspection, icon=":material/restart_alt:")
        return
    stage = draft["stage"]
    nonce = draft["nonce"]
    labels = ["Bâtiment", "Photo", "Repère", "Mesures", "Résumé"]
    st.html('<div class="steps">'+"".join(
        f'<div class="step {"active" if i == stage else "done" if i < stage else ""}">{i+1}. {label}</div>'
        for i, label in enumerate(labels))+'</div>')
    buildings = {b["id"]: b for b in data["buildings"]}
    b = buildings.get(draft.get("building_id"))

    def advance():
        draft["stage"] += 1
        st.rerun()

    if stage == 0:
        options = ["NEW"] + list(buildings)
        chosen = draft.get("building_id") if draft.get("building_id") in buildings else "NEW"
        bid = st.selectbox("Bâtiment à inspecter", options, index=options.index(chosen),
                           format_func=lambda v: "Créer un bâtiment" if v == "NEW" else buildings[v]["name"], key="wizard_b_"+nonce)
        with st.form("wizard_building_"+nonce):
            if bid == "NEW":
                name = st.text_input("Nom du bâtiment", draft.get("building_name", ""), max_chars=100)
                address = st.text_input("Emplacement", draft.get("address", ""), max_chars=200)
            else:
                name, address = buildings[bid]["name"], buildings[bid]["address"]
                st.write(name)
                st.caption(address or "Emplacement non renseigné")
            if st.form_submit_button("Continuer vers la photo", type="primary", icon=":material/arrow_forward:"):
                try:
                    name = text(name, 100, "Bâtiment", True)
                    address = text(address, 200, "Emplacement")
                    if bid == "NEW" and name.casefold() in {bb["name"].casefold() for bb in buildings.values()}:
                        raise ValueError("Ce nom de bâtiment existe déjà.")
                    new_bid = (draft.get("building_id") if draft.get("new_building") else uid()) if bid == "NEW" else bid
                    if new_bid != draft.get("building_id"):
                        for k in ("crack_id", "new_crack", "reference_photo", "visit_photo", "code", "location", "x", "y", "next_visit"):
                            draft.pop(k, None)
                    draft.update(building_id=new_bid, new_building=bid == "NEW", building_name=name, address=address)
                    advance()
                except ValueError as ex:
                    st.error(str(ex))
    elif stage == 1:
        left, right = st.columns([1.5, 1], gap="large")
        with right:
            st.subheader("Photo de l’inspection")
            with st.form("wizard_photo_"+nonce):
                uploaded = st.file_uploader("Photo JPEG / PNG", type=["jpg", "jpeg", "png"], max_upload_size=5)
                clear = st.checkbox("Continuer sans photo de visite")
                st.caption("La photo est facultative. Pour un bâtiment existant, sa vue de référence reste inchangée.")
                if st.form_submit_button("Continuer vers le repère", type="primary", icon=":material/arrow_forward:"):
                    try:
                        photo = normalize_photo(uploaded.getvalue()) if uploaded else ("" if clear else draft.get("visit_photo", ""))
                        draft["visit_photo"] = photo
                        draft["reference_photo"] = photo if draft["new_building"] else b["photo"]
                        advance()
                    except ValueError as ex:
                        st.error(str(ex))
        with left:
            show_photo(draft.get("visit_photo") or (b["photo"] if b else ""), key="wizard_preview_"+nonce,
                       caption="Photo conservée dans le brouillon" if draft.get("visit_photo") else "Vue de référence existante")
    elif stage == 2:
        original = {c["id"]: c for c in b["cracks"]} if b else {}
        options = ["NEW"] + list(original)
        chosen = draft.get("crack_id") if draft.get("crack_id") in original else "NEW"
        cid = st.selectbox("Repère à inspecter", options, index=options.index(chosen),
                           format_func=lambda v: "Créer un repère" if v == "NEW" else original[v]["code"]+" · "+original[v]["location"],
                           key="wizard_c_"+nonce+draft["building_id"])
        left, right = st.columns([1.5, 1], gap="large")
        with left:
            point = dict(x=draft.get("x", 50.), y=draft.get("y", 50.))
            result = show_photo(draft.get("reference_photo", ""), list(original.values()), edit=cid == "NEW", point=point if cid == "NEW" else None,
                                key="wizard_pin_"+nonce, selected=cid, caption="Repérage sur la vue de référence")
            if result.point and cid == "NEW":
                point = {"x": number(result.point["x"], 0, 100, "X"), "y": number(result.point["y"], 0, 100, "Y")}
                draft.update(point)
                st.session_state["wx_"+nonce], st.session_state["wy_"+nonce] = point["x"], point["y"]
        with right:
            with st.form("wizard_marker_"+nonce):
                if cid == "NEW":
                    code = st.text_input("Code du repère", draft.get("code", ""), max_chars=30, placeholder="F-03")
                    loc = st.text_input("Emplacement de la fissure", draft.get("location", ""), max_chars=200)
                    st.session_state.setdefault("wx_"+nonce, point["x"])
                    st.session_state.setdefault("wy_"+nonce, point["y"])
                    x = st.number_input("X (%)", 0., 100., step=.1, key="wx_"+nonce)
                    y = st.number_input("Y (%)", 0., 100., step=.1, key="wy_"+nonce)
                else:
                    cc = original[cid]
                    code, loc, x, y = cc["code"], cc["location"], cc["x"], cc["y"]
                    st.write(code)
                    st.caption(loc)
                if st.form_submit_button("Continuer vers les mesures", type="primary", icon=":material/arrow_forward:"):
                    try:
                        code = text(code, 30, "Repère", True)
                        if cid == "NEW" and code.casefold() in {cc["code"].casefold() for cc in original.values()}:
                            raise ValueError("Ce code de repère est déjà utilisé.")
                        new_cid = (draft.get("crack_id") if draft.get("new_crack") else uid()) if cid == "NEW" else cid
                        draft.update(crack_id=new_cid, new_crack=cid == "NEW", code=code, location=loc, x=x, y=y)
                        advance()
                    except ValueError as ex:
                        st.error(str(ex))
    elif stage == 3:
        previous = draft.get("visit", {})
        original = next((c for c in (b["cracks"] if b else []) if c["id"] == draft["crack_id"]), None)
        planned = draft.get("next_visit", original.get("next_visit", "") if original else "")
        with st.form("wizard_readings_"+nonce):
            st.subheader(draft["building_name"]+" / "+draft["code"])
            day = st.date_input("Date de visite", date.fromisoformat(previous["date"]) if previous else today(), min_value=date(1900, 1, 1), max_value=today())
            readings = st.text_area("Largeurs mesurées (mm)", "; ".join(f"{v:g}" for v in previous.get("readings", [])),
                                    placeholder="0,29 ; 0,30 ; 0,31", max_chars=3000)
            u = st.number_input("Incertitude-type instrumentale (mm)", 0., 10., previous.get("u", .02), step=.005, format="%.3f")
            notes = st.text_area("Observations", previous.get("notes", ""), max_chars=2000)
            active = st.checkbox("Planifier une prochaine visite", value=bool(planned))
            due = st.date_input("Date prévue", date.fromisoformat(planned) if planned else today()+timedelta(days=14),
                                min_value=date(1900, 1, 1), max_value=date(2100, 12, 31))
            if st.form_submit_button("Vérifier l’inspection", type="primary", icon=":material/checklist:"):
                try:
                    if original and any(v["date"] == day.isoformat() for v in original["visits"]):
                        raise ValueError("Une visite existe déjà à cette date. Modifie-la depuis le Carnet de visites.")
                    draft["visit"] = dict(id=previous.get("id", uid()), date=day.isoformat(), readings=parse_readings(readings),
                                          u=u, notes=notes, photo=draft.get("visit_photo", ""))
                    draft["next_visit"] = due.isoformat() if active else ""
                    build_inspection(data, draft)
                    advance()
                except ValueError as ex:
                    st.error(str(ex))
    else:
        v = draft["visit"]
        s = visit_stats(v)
        st.subheader("Inspection prête à enregistrer")
        st.table(pd.DataFrame({"Champ": ["Bâtiment", "Repère", "Date", "Moyenne", "Répétitions", "Prochaine visite"],
                               "Valeur": [draft["building_name"], draft["code"], date_label(v["date"]), fmt(s["mean"], 3)+" mm", str(s["n"]), date_label(draft["next_visit"])]}),
                 alt="Résumé de l’inspection avant enregistrement.")
        st.write(v["notes"] or "Sans observation écrite.")
        if v["photo"]:
            st.image(BytesIO(base64.b64decode(v["photo"])), width=450, alt="Photo de l’inspection à enregistrer.")
        if st.button("Enregistrer l’inspection", type="primary", icon=":material/save:", key="wizard_save"):
            try:
                new = build_inspection(data, draft)
                st.session_state.selected_building, st.session_state.selected_crack = draft["building_id"], draft["crack_id"]
                st.session_state.pop("building_picker", None)
                st.session_state.pop("crack_picker_"+draft["building_id"], None)
                persist(new, rev, next_page="Fiches fissures")
            except ValueError as ex:
                st.error(str(ex))
        st.caption("Rien n’est ajouté au registre avant l’enregistrement final.")
    if stage > 0 and st.button("Étape précédente", icon=":material/arrow_back:", key="wizard_back_"+nonce):
        draft["stage"] -= 1
        st.rerun()


def buildings_page(data, rev):
    with st.expander("Nouveau bâtiment", expanded=not data["buildings"], icon=":material/add:"):
        with st.form("new_building"):
            name = st.text_input("Nom", max_chars=100)
            address = st.text_input("Localisation du bâtiment", max_chars=200)
            upload = st.file_uploader("Photo de référence (JPEG/PNG, 5 Mo maximum)", type=["jpg", "jpeg", "png"], max_upload_size=5)
            if st.form_submit_button("Ajouter le bâtiment", icon=":material/add:", type="primary"):
                try:
                    new = deepcopy(data)
                    new["buildings"].append(dict(id=uid(), name=name, address=address,
                        photo=normalize_photo(upload.getvalue()) if upload else "", cracks=[]))
                    persist(new, rev)
                except ValueError as e:
                    st.error(str(e))
    b = select_building(data)
    if not b:
        return
    left, right = st.columns([1.6, 1], gap="large")
    key = f"{st.session_state.space}_{b['id']}_{rev}"
    with left:
        st.subheader("Plan de repérage")
        point = st.session_state.get("point_"+b["id"], {"x": 50., "y": 50.})
        result = show_photo(b["photo"], b["cracks"], key="edit_plate_"+key, edit=True, point=point,
                            caption="Photo de référence et repères enregistrés.")
        if result.point:
            p = result.point
            point = {"x": number(p["x"], 0, 100, "X"), "y": number(p["y"], 0, 100, "Y")}
            st.session_state["point_"+b["id"]] = point
            st.session_state["px_"+key] = point["x"]
            st.session_state["py_"+key] = point["y"]
        if b["cracks"]:
            st.dataframe(pd.DataFrame([dict(Repère=c["code"], Localisation=c["location"], Visites=len(c["visits"])) for c in b["cracks"]]), hide_index=True, alt="Repères du bâtiment.")
    with right:
        st.subheader("Ajouter un repère")
        st.caption("Point sélectionné sur l’image, ou coordonnées en % depuis le coin supérieur gauche.")
        with st.form("crack_"+key):
            code = st.text_input("Code du repère", max_chars=30, placeholder="F-03")
            loc = st.text_input("Emplacement de la fissure", max_chars=200)
            xy = st.columns(2)
            st.session_state.setdefault("px_"+key, point["x"])
            st.session_state.setdefault("py_"+key, point["y"])
            x = xy[0].number_input("X (%)", 0., 100., step=.1, key="px_"+key)
            y = xy[1].number_input("Y (%)", 0., 100., step=.1, key="py_"+key)
            if st.form_submit_button("Enregistrer le repère", icon=":material/add_location_alt:", type="primary"):
                new = deepcopy(data)
                bb = next(bb for bb in new["buildings"] if bb["id"] == b["id"])
                bb["cracks"].append(dict(id=uid(), code=code, location=loc, x=x, y=y, visits=[]))
                persist(new, rev)
    with st.expander("Modifier le dossier ou la photo", icon=":material/edit:"):
        with st.form("edit_b_"+key):
            name = st.text_input("Nom du dossier", b["name"], max_chars=100)
            address = st.text_input("Adresse / emplacement", b["address"], max_chars=200)
            photo = st.file_uploader("Nouvelle photo de référence", type=["jpg", "jpeg", "png"], max_upload_size=5)
            clear = st.checkbox("Retirer la photo actuelle")
            st.caption("Après un changement de cadrage, repositionne les repères manuellement.")
            if st.form_submit_button("Enregistrer le dossier", icon=":material/save:"):
                try:
                    new = deepcopy(data)
                    bb = next(bb for bb in new["buildings"] if bb["id"] == b["id"])
                    bb.update(name=name, address=address)
                    bb["photo"] = normalize_photo(photo.getvalue()) if photo else ("" if clear else b["photo"])
                    persist(new, rev)
                except ValueError as e:
                    st.error(str(e))
    if b["cracks"]:
        with st.expander("Modifier ou supprimer un repère", icon=":material/edit_location_alt:"):
            c = select_crack(b)
            with st.form("edit_c_"+key+c["id"]):
                code = st.text_input("Repère", c["code"], max_chars=30)
                loc = st.text_input("Localisation", c["location"], max_chars=200)
                x = st.number_input("Position horizontale (%)", 0., 100., c["x"], step=.1)
                y = st.number_input("Position verticale (%)", 0., 100., c["y"], step=.1)
                if st.form_submit_button("Modifier le repère", icon=":material/save:"):
                    new = deepcopy(data)
                    cc = next(cc for bb in new["buildings"] for cc in bb["cracks"] if cc["id"] == c["id"])
                    cc.update(code=code, location=loc, x=x, y=y)
                    persist(new, rev)
            confirm = st.checkbox("Supprimer ce repère et toutes ses visites", key="dc_"+key+c["id"])
            if st.button("Supprimer le repère", icon=":material/delete:", disabled=not confirm):
                new = deepcopy(data)
                bb = next(bb for bb in new["buildings"] if bb["id"] == b["id"])
                bb["cracks"] = [cc for cc in bb["cracks"] if cc["id"] != c["id"]]
                persist(new, rev)
    with st.expander("Supprimer le bâtiment", icon=":material/delete:"):
        confirm = st.checkbox("Supprimer le bâtiment, ses repères, ses visites et ses photos", key="db_"+key)
        if st.button("Confirmer la suppression du bâtiment", icon=":material/delete_forever:", disabled=not confirm):
            new = deepcopy(data)
            new["buildings"] = [bb for bb in new["buildings"] if bb["id"] != b["id"]]
            persist(new, rev)


def replace_crack(data, c):
    new = deepcopy(data)
    for b in new["buildings"]:
        for i, cc in enumerate(b["cracks"]):
            if cc["id"] == c["id"]:
                b["cracks"][i] = c
                return new
    raise ValueError("Repère introuvable.")


def visits_page(data, rev):
    b = select_building(data)
    if not b:
        return
    c = select_crack(b)
    if not c:
        return
    vs = sorted(c["visits"], key=lambda v: v["date"])
    e = evolution(c)
    strip([("Dernière largeur", fmt(visit_stats(vs[-1])["mean"])+" mm" if vs else "--", "Moyenne de la visite"),
           ("Variation observée", fmt(e["delta"])+" mm" if e else "--", "Dernière moins première"),
           ("Variation / jour", fmt(e["rate"], 3) if e else "--", "mm/j sur la période passée"),
           ("Visites", str(len(vs)).zfill(2), "Une fiche par date")])
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        if vs:
            st.plotly_chart(evolution_chart([c]), theme=None, alt="Évolution des largeurs observées et barres d’incertitude.")
        else:
            show_photo(b["photo"], [c], key="visit_empty_"+c["id"])
        st.subheader("Historique des visites")
        if vs:
            st.html('<div class="timeline">'+"".join(
                f'<div class="entry"><time>{v["date"]} · {len(v["readings"])} mesure(s)</time><strong>{fmt(visit_stats(v)["mean"])} mm</strong><p>{escape(v["notes"])}</p></div>'
                for v in reversed(vs))+'</div>')
    with right:
        st.subheader("Consigner une visite")
        mode = st.segmented_control("Opération", ["Ajouter", "Modifier"], default="Ajouter", key="visit_mode")
        existing = None
        if mode == "Modifier":
            if not vs:
                st.caption("Aucune visite à modifier.")
                return
            vid = st.selectbox("Visite à modifier", [v["id"] for v in vs], format_func=lambda x: next(v["date"] for v in vs if v["id"] == x))
            existing = next(v for v in vs if v["id"] == vid)
        key = f"{st.session_state.space}_{c['id']}_{rev}_{existing['id'] if existing else 'new'}"
        with st.form("visit_form_"+key):
            day = st.date_input("Date de visite", date.fromisoformat(existing["date"]) if existing else today(), min_value=date(1900, 1, 1), max_value=today())
            readings = st.text_area("Largeurs mesurées (mm)", "; ".join(f"{x:g}" for x in existing["readings"]) if existing else "",
                                   placeholder="0,29 ; 0,30 ; 0,31", max_chars=3000,
                                   help="Même point, même visite. Point-virgule entre les mesures ; virgule décimale acceptée.")
            u = st.number_input("Incertitude-type instrumentale (mm)", 0., 10., existing["u"] if existing else .02, step=.005, format="%.3f",
                                help="Hypothèse sur la moyenne de cette visite, pas une tolérance réglementaire.")
            notes = st.text_area("Observations", existing["notes"] if existing else "", max_chars=2000)
            photo = st.file_uploader("Photo de cette visite", type=["jpg", "jpeg", "png"], max_upload_size=5)
            remove_photo = st.checkbox("Retirer la photo de cette visite") if existing else False
            if st.form_submit_button("Enregistrer la visite", icon=":material/save:", type="primary"):
                try:
                    v = dict(id=existing["id"] if existing else uid(), date=day.isoformat(), readings=parse_readings(readings),
                             u=u, notes=notes, photo=normalize_photo(photo.getvalue()) if photo else
                             ("" if remove_photo else (existing["photo"] if existing else "")))
                    newc = deepcopy(c)
                    newc["visits"] = [vv for vv in newc["visits"] if not existing or vv["id"] != existing["id"]] + [v]
                    persist(replace_crack(data, newc), rev)
                except ValueError as ex:
                    st.error(str(ex))
        if existing:
            confirm = st.checkbox("Supprimer cette visite et sa photo", key="dv_"+key)
            if st.button("Supprimer la visite", icon=":material/delete:", disabled=not confirm):
                newc = deepcopy(c)
                newc["visits"] = [v for v in vs if v["id"] != existing["id"]]
                persist(replace_crack(data, newc), rev)
    with st.expander("Importer des visites CSV / Excel", icon=":material/upload_file:"):
        st.caption("Ajout au repère sélectionné, sans écraser ses visites. Les dates déjà présentes sont refusées.")
        template = pd.DataFrame([dict(date=(today()-timedelta(days=1)).isoformat(), mesures_mm="0.29;0.30;0.31", u_instrument_mm=.02, notes="Exemple fictif à adapter")])
        st.download_button("Modèle CSV", csv_bytes(template), "modele_visites.csv", "text/csv", icon=":material/download:", on_click="ignore")
        with st.form("import_"+key):
            file = st.file_uploader("Fichier de visites", type=["csv", "xlsx"], max_upload_size=2)
            if st.form_submit_button("Importer dans ce repère", icon=":material/upload:"):
                if not file:
                    st.error("Choisis un fichier.")
                else:
                    try:
                        newc = import_visits(file.getvalue(), file.name.rsplit(".", 1)[-1], c)
                        persist(replace_crack(data, newc), rev)
                    except (ValueError, OSError, KeyError) as ex:
                        st.error(str(ex))


def compare_page(data, rev):
    b = select_building(data)
    if not b:
        return
    c = select_crack(b)
    if not c:
        return
    vs = sorted([v for v in c["visits"] if v["photo"]], key=lambda v: v["date"])
    if len(vs) < 2:
        st.info("Il faut deux visites avec une photo pour ouvrir le comparateur.")
        return
    cols = st.columns(2)
    ka, kb = "compare_a_"+c["id"], "compare_b_"+c["id"]
    if st.session_state.get(ka, 0) >= len(vs):
        st.session_state.pop(ka, None)
    if st.session_state.get(kb, 0) >= len(vs):
        st.session_state.pop(kb, None)
    a = cols[0].selectbox("Photo initiale", range(len(vs)), index=0, format_func=lambda i: vs[i]["date"], key=ka)
    bidx = cols[1].selectbox("Photo finale", range(len(vs)), index=0 if kb in st.session_state else len(vs)-1,
                            format_func=lambda i: vs[i]["date"], key=kb)
    if vs[a]["date"] >= vs[bidx]["date"]:
        st.warning("Choisis une photo finale postérieure à la photo initiale.")
    else:
        show_photo(vs[a]["photo"], compare=vs[bidx]["photo"], key=f"compare_{c['id']}_{a}_{bidx}",
                   caption=f"Gauche : {date_label(vs[a]['date'])} · Droite : {date_label(vs[bidx]['date'])}")
        delta = visit_stats(vs[bidx])["mean"]-visit_stats(vs[a])["mean"]
        note(f"Variation des mesures saisies : {fmt(delta, 3)} mm. Les photos ne servent pas à calculer la largeur.")
        st.caption("Zoom et déplacement communs aux deux vues. Cadrages différents : pas de recalage automatique, la comparaison reste indicative.")
    st.subheader("Choisir parmi les visites photographiées")
    if len(vs) > 12:
        st.caption("Les 12 dernières miniatures sont affichées ; toutes les dates restent accessibles dans les sélecteurs.")
    cols = st.columns(3)
    for i in range(max(0, len(vs)-12), len(vs)):
        with cols[i % 3]:
            thumb(vs[i]["photo"], "Photo du "+vs[i]["date"])
            st.caption(f"{date_label(vs[i]['date'])} · {fmt(visit_stats(vs[i])['mean'])} mm")
            with st.container(horizontal=True):
                st.button("Avant", icon=":material/first_page:", key="before_"+vs[i]["id"],
                          on_click=lambda k=ka, idx=i: st.session_state.update({k:idx}))
                st.button("Après", icon=":material/last_page:", key="after_"+vs[i]["id"],
                          on_click=lambda k=kb, idx=i: st.session_state.update({k:idx}))


def uncertainty_page(data, rev):
    b = select_building(data)
    if not b:
        return
    c = select_crack(b)
    if not c:
        return
    vs = sorted(c["visits"], key=lambda v: v["date"])
    if len(vs) < 2:
        st.info("Enregistre au moins deux visites pour comparer les incertitudes.")
        return
    with st.form("uncertainty_"+c["id"]):
        cols = st.columns(2)
        a = cols[0].selectbox("Visite initiale", range(len(vs)), index=0, format_func=lambda i: vs[i]["date"])
        bi = cols[1].selectbox("Visite finale", range(len(vs)), index=len(vs)-1, format_func=lambda i: vs[i]["date"])
        threshold = st.number_input("Variation de comparaison (mm)", 0., 10., .10, step=.01,
                                    help="Seuil libre d’étude, pas un seuil de danger.")
        seed = st.number_input("Graine de simulation", 0, 1_000_000, 42)
        run = st.form_submit_button("Étudier les incertitudes", type="primary", icon=":material/query_stats:")
    signature = (st.session_state.space, c["id"], rev)
    if run:
        try:
            result = uncertainty(vs[a], vs[bi], threshold, seed=int(seed))
            st.session_state.uncertainty_result = (signature, (a, bi, threshold, seed, result))
        except ValueError as ex:
            st.error(str(ex))
    latest = st.session_state.get("uncertainty_result")
    stored = latest[1] if latest and latest[0] == signature else None
    if not stored:
        st.caption("Sélectionne les deux visites puis lance l’étude.")
    else:
        a, bi, threshold, seed, result = stored
        strip([("Différence des moyennes", fmt(result["delta"], 3)+" mm", "Finale moins initiale"),
               ("Incertitude-type Δ", fmt(result["u"], 3)+" mm", "Erreurs supposées indépendantes"),
               ("P(Δ > seuil)", fmt(100*result["p"], 1)+" %", "Loi normale supposée"),
               ("Monte-Carlo", fmt(100*result["simulated_p"], 1)+" %", "20 000 tirages simulés")])
        left, right = st.columns([1.6, 1], gap="large")
        with left:
            fig = go.Figure(go.Histogram(x=result["samples"], nbinsx=55, marker_color=TEAL,
                                         histnorm="probability density", name="Tirages simulés"))
            fig.add_vline(x=threshold, line_color=CORAL, line_dash="dash")
            fig.update_xaxes(title="Variation simulée (mm)")
            fig.update_yaxes(title="Densité")
            st.plotly_chart(chart_layout(fig), theme=None, alt="Distribution simulée de la différence des mesures et seuil de comparaison.")
        with right:
            st.subheader("Lecture du résultat")
            st.write(f"Comparaison : {vs[a]['date']} → {vs[bi]['date']}.")
            st.write(f"Intervalle normal approximatif à 95 % : [{fmt(result['low'], 3)} ; {fmt(result['high'], 3)}] mm.")
            st.write("Le zéro est " + ("hors de cet intervalle." if result["low"] > 0 or result["high"] < 0 else "dans cet intervalle."))
            note("Ce pourcentage décrit un modèle d’incertitude sur une différence de mesures. Ce n’est ni un risque d’effondrement, ni une prédiction de la fissure.")
        st.download_button("Tirages simulés CSV", csv_bytes(pd.DataFrame({"variation_simulee_mm": result["samples"]})),
                           "simulation_incertitude.csv", "text/csv", icon=":material/download:", on_click="ignore")
    st.subheader("Hypothèses et formules")
    st.latex(r"\bar{x}=\frac{1}{n}\sum x_i")
    st.latex(r"u_{visite}=\sqrt{u_{instrument}^2+\frac{s^2}{n}}")
    st.latex(r"\Delta=\bar{x}_{finale}-\bar{x}_{initiale}")
    st.latex(r"u_\Delta=\sqrt{u_{initiale}^2+u_{finale}^2}")
    st.write("s est l’écart-type des répétitions (n−1 au dénominateur). Avec une seule mesure, la dispersion n’est pas estimable : seule l’incertitude instrumentale saisie est retenue.")
    st.write("Le modèle suppose des erreurs centrées, normales et indépendantes entre visites. Un biais commun d’instrument, la température, le changement de point ou d’opérateur ne sont pas modélisés. L’intervalle ±1,96 u est approximatif, surtout avec peu de répétitions. Les tirages peuvent être négatifs car ils décrivent une différence, pas une largeur.")
    st.caption("Si toutes les incertitudes sont nulles, le calcul est déterministe ; ce choix ne signifie pas qu’un appareil réel est parfait.")
    frame = visits_frame(b)
    st.dataframe(frame[frame["repere"] == c["code"]].drop(columns=["batiment", "notes"]), hide_index=True,
                 alt="Statistiques descriptives des mesures observées par visite.")


def excel_bytes(b):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        safe_frame(visits_frame(b)).to_excel(writer, index=False, sheet_name="Visites")
        safe_frame(pd.DataFrame([dict(repere=c["code"], localisation=c["location"], x_pct=c["x"], y_pct=c["y"],
            prochaine_visite=c.get("next_visit", ""), suivi=followup(c)["label"]) for c in b["cracks"]])).to_excel(writer, index=False, sheet_name="Reperes")
    return out.getvalue()


class ReferencePlate(Flowable):
    """Photo and vector markers; positions refer only to this reference view."""
    def __init__(self, photo, cracks, font, selected=None):
        super().__init__()
        self.raw = base64.b64decode(photo)
        with Image.open(BytesIO(self.raw)) as im:
            w, h = im.size
        scale = min(510/w, 235/h)
        self.width, self.height = w*scale, h*scale
        self.cracks, self.font, self.selected = cracks, font, selected

    def draw(self):
        can = self.canv
        can.drawImage(ImageReader(BytesIO(self.raw)), 0, 0, self.width, self.height)
        for c in self.cracks:
            x, y = self.width*c["x"]/100, self.height*(1-c["y"]/100)
            can.setFillColor(colors.HexColor(CORAL if c["id"] == self.selected else INK))
            can.setStrokeColor(colors.white)
            can.circle(x, y, 4, fill=1, stroke=1)
            bw = min(self.width, pdfmetrics.stringWidth(c["code"], self.font, 7)+12)
            bx = max(0, min(self.width-bw, x-bw/2))
            by = max(0, min(self.height-16, y+7))
            can.roundRect(bx, by, bw, 16, 3, fill=1, stroke=0)
            can.setFillColor(colors.white)
            can.setFont(self.font, 7)
            can.drawCentredString(bx+bw/2, by+5, c["code"])


class EvolutionPlot(Flowable):
    """Vector PDF chart, requiring no browser or image service."""
    def __init__(self, crack, font):
        super().__init__()
        self.width, self.height = 510, 180
        self.visits = sorted(crack["visits"], key=lambda v: v["date"])
        self.font = font

    def draw(self):
        can = self.canv
        pts = [(date.fromisoformat(v["date"]).toordinal(), visit_stats(v)) for v in self.visits]
        if not pts:
            return
        xmin, xmax = pts[0][0], pts[-1][0]
        lo = min(s["mean"]-1.96*s["u"] for _, s in pts)
        hi = max(s["mean"]+1.96*s["u"] for _, s in pts)
        pad = max((hi-lo)*.18, .02)
        lo, hi = lo-pad, hi+pad
        xy = lambda d, y: (52+(d-xmin)/max(1, xmax-xmin)*440, 34+(y-lo)/(hi-lo)*122)
        can.setFont(self.font, 8)
        for y in np.linspace(lo, hi, 4):
            _, yy = xy(xmin, y)
            can.setStrokeColor(colors.HexColor("#DEE2E8"))
            can.line(52, yy, 492, yy)
            can.setFillColor(colors.HexColor(INK))
            can.drawRightString(45, yy-3, fmt(y, 3))
        can.drawString(4, 167, "mm")
        can.setStrokeColor(colors.HexColor(CORAL))
        can.setLineWidth(1.6)
        if len(pts) > 1:
            path = can.beginPath()
            path.moveTo(*xy(pts[0][0], pts[0][1]["mean"]))
            for d, s in pts[1:]:
                path.lineTo(*xy(d, s["mean"]))
            can.drawPath(path)
        for d, s in pts:
            x, y = xy(d, s["mean"])
            low, high = xy(d, s["mean"]-1.96*s["u"])[1], xy(d, s["mean"]+1.96*s["u"])[1]
            can.setStrokeColor(colors.HexColor(TEAL))
            can.setLineWidth(.7)
            can.line(x, low, x, high)
            can.line(x-3, low, x+3, low)
            can.line(x-3, high, x+3, high)
            can.setFillColor(colors.HexColor(CORAL))
            can.circle(x, y, 2.4, fill=1, stroke=0)
        can.setFillColor(colors.HexColor(INK))
        can.drawString(52, 17, date_label(self.visits[0]["date"]))
        if len(pts) > 1:
            can.drawRightString(492, 17, date_label(self.visits[-1]["date"]))


def pdf_bytes(b, fictive=False):
    regular, bold = "Helvetica", "Helvetica-Bold"
    for r, bd in [("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
                  ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]:
        if Path(r).exists() and Path(bd).exists():
            if "FSRegular" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("FSRegular", r))
                pdfmetrics.registerFont(TTFont("FSBold", bd))
            regular, bold = "FSRegular", "FSBold"
            break
    styles = getSampleStyleSheet()
    body = ParagraphStyle("FSBody", fontName=regular, fontSize=9, leading=13, spaceAfter=7, wordWrap="CJK")
    title = ParagraphStyle("FSTitle", parent=body, fontName=bold, fontSize=23, leading=28, spaceAfter=12)
    heading = ParagraphStyle("FSHeading", parent=body, fontName=bold, fontSize=13, leading=18, spaceBefore=12)
    small = ParagraphStyle("FSSmall", parent=body, fontSize=8, leading=11)
    p = lambda s, style=body: Paragraph(escape(str(s)), style)
    out = BytesIO()
    doc = SimpleDocTemplate(out, pagesize=(595.28, 841.89), leftMargin=42, rightMargin=42, topMargin=44, bottomMargin=44)
    story = [p("FissureScope", title), p("Carnet d’inspection - " + b["name"], heading),
             p(b["address"]), p(f"Édité le {today().strftime('%d/%m/%Y')} · " + ("EXEMPLE FICTIF" if fictive else "DONNÉES SAISIES")),
             p("Suivi pédagogique des observations. Aucun diagnostic de cause, de gravité ou de sécurité du bâtiment.")]
    if b["photo"]:
        story += [ReferencePlate(b["photo"], b["cracks"], bold), Spacer(1, 9),
                  p("Vue de référence annotée. Les positions sont saisies par l’utilisateur ; elles ne mesurent pas les largeurs.", small)]
    story.append(p(f"{len(b['cracks'])} repères · {sum(len(c['visits']) for c in b['cracks'])} visites · "
                   f"{building_followup(b)['late']} visite(s) en retard."))
    story.append(p("Les statuts portent sur le calendrier et la présence de relevés, jamais sur la sécurité du bâtiment.", small))
    for c in b["cracks"]:
        e = evolution(c)
        story += [PageBreak(), KeepTogether([p(c["code"] + " - " + c["location"], heading),
                    p(f"Position sur la vue : X={fmt(c['x'], 1)} % ; Y={fmt(c['y'], 1)} %.")])]
        if e:
            story.append(p(f"Variation observée : {fmt(e['delta'], 3)} mm en {e['days']} jours ; moyenne passée {fmt(e['rate'], 4)} mm/j. Ce taux n’est pas une prévision."))
        status = followup(c)
        story.append(p(f"Suivi : {status['label']}. Prochaine visite à effectuer : {date_label(status['due'])}.", small))
        if c["visits"]:
            story += [EvolutionPlot(c, regular),
                      p("Évolution des moyennes observées. Barres : ±1,96 u visite, approximation normale. Le trait relie les observations, sans prévision.", small)]
        photos = sorted([v for v in c["visits"] if v["photo"]], key=lambda v: v["date"])
        if photos:
            chosen = [photos[0]] if len(photos) == 1 else [photos[0], photos[-1]]
            cells = []
            for i, v in enumerate(chosen):
                raw = BytesIO(base64.b64decode(v["photo"]))
                with Image.open(raw) as im:
                    w, h = im.size
                raw.seek(0)
                scale = min(242/w, 148/h)
                label = "Première photo" if i == 0 else "Dernière photo"
                cells.append([p(label + " · " + date_label(v["date"]), small),
                              PDFImage(raw, width=w*scale, height=h*scale)])
            table_photos = Table([cells], colWidths=[255]*len(cells), hAlign="LEFT")
            table_photos.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 10)]))
            story += [table_photos, p("Photos non recalées : la perspective et le cadrage peuvent différer. Aucune largeur n’est calculée sur ces images.", small)]
        rows = [[p(x, small) for x in ["Date", "Moyenne (mm)", "n", "u visite (mm)", "Observations"]]]
        for v in sorted(c["visits"], key=lambda v: v["date"]):
            s = visit_stats(v)
            rows.append([p(v["date"], small), p(fmt(s["mean"], 3), small), p(s["n"], small),
                         p(fmt(s["u"], 3), small), p(v["notes"], small)])
        if len(rows) > 1:
            table = Table(rows, colWidths=[68, 90, 25, 83, 244], repeatRows=1, splitInRow=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(LIME)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#CBD2C8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
            story += [table, Spacer(1, 10)]
        else:
            story.append(p("Aucune visite enregistrée."))
    story += [p("Méthode et limites", heading), p("Les largeurs sont saisies par l’utilisateur, jamais déduites automatiquement d’une photo. La moyenne et l’écart-type décrivent les répétitions d’une visite. u visite = racine(u instrument² + s²/n), avec s non estimable si n = 1. Les barres ±1,96u supposent un modèle normal approximatif. Les causes, les biais communs et la sécurité de l’ouvrage ne sont pas évalués."),
              p("Ce rapport contient la vue annotée, les graphiques, les mesures et notes, ainsi que la première et la dernière photo disponibles par repère. Le JSON conserve toutes les photos ; les tirages simulés sont exportés séparément en CSV.", small)]

    def footer(can, _):
        can.setStrokeColor(colors.HexColor(INK))
        can.line(42, 35, 553, 35)
        can.setFont(regular, 8)
        can.drawString(42, 22, "FissureScope · suivi pédagogique")
        can.drawRightString(553, 22, str(doc.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()


def backup_bytes(data):
    return json.dumps(validate(data), ensure_ascii=False, indent=2).encode("utf-8")


def restore_copies(current, raw):
    if len(raw) > 25_000_000:
        raise ValueError("Sauvegarde trop volumineuse : maximum 25 Mo.")
    try:
        incoming = validate(json.loads(raw))
    except (json.JSONDecodeError, UnicodeError, RecursionError):
        raise ValueError("Sauvegarde JSON illisible.") from None
    out = deepcopy(current)
    names = {b["name"].casefold() for b in out["buildings"]}
    for b in incoming["buildings"]:
        base = b["name"][:75]
        name, i = base, 1
        while name.casefold() in names:
            name = f"{base} (copie {i})"
            i += 1
        b["name"], b["id"] = name, uid()
        names.add(name.casefold())
        for c in b["cracks"]:
            c["id"] = uid()
            for v in c["visits"]:
                v["id"] = uid()
        out["buildings"].append(b)
    return validate(out)


def reports_page(data, rev):
    b = select_building(data)
    if b:
        st.subheader("Dossier du bâtiment")
        with st.container(horizontal=True):
            st.download_button("Rapport PDF", pdf_bytes(b, st.session_state.space == "Exemple fictif"), "FissureScope_rapport.pdf", "application/pdf", icon=":material/picture_as_pdf:", on_click="ignore")
            st.download_button("Mesures CSV", csv_bytes(visits_frame(b)), "FissureScope_mesures.csv", "text/csv", icon=":material/table_view:", on_click="ignore")
            st.download_button("Classeur Excel", excel_bytes(b), "FissureScope_mesures.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon=":material/download:", on_click="ignore")
        st.dataframe(visits_frame(b), hide_index=True, alt="Mesures observées du bâtiment exporté.")
    st.subheader("Sauvegarde de tout l’espace")
    st.download_button("Sauvegarde complète JSON", backup_bytes(data), "FissureScope_sauvegarde.json", "application/json", icon=":material/save:", on_click="ignore")
    st.caption("Le JSON conserve bâtiments, repères, mesures, notes et photos. L’exemple fictif et tes données sont exportés séparément.")
    with st.form("restore"):
        upload = st.file_uploader("Restaurer une sauvegarde FissureScope", type=["json"], max_upload_size=25)
        confirm = st.checkbox("Ajouter les bâtiments de la sauvegarde comme copies, sans supprimer les dossiers existants")
        if st.form_submit_button("Importer les copies", icon=":material/upload:"):
            if not upload or not confirm:
                st.error("Choisis une sauvegarde et confirme l’ajout de copies.")
            else:
                try:
                    persist(restore_copies(data, upload.getvalue()), rev)
                except ValueError as ex:
                    st.error(str(ex))
    note("En local, tes données sont enregistrées dans SQLite, hors du dossier de l’application. Sur un hébergement public, le registre est commun aux visiteurs et le disque peut être temporaire. Utilise des données fictives et conserve tes sauvegardes JSON.")


def teaching_page(data, rev):
    st.subheader("Ce que FissureScope fait vraiment")
    st.write("Tu photographies un mur, repères une fissure puis reviens mesurer le même point. Le carnet te montre les mesures datées, leur variation et leur incertitude. Il ne dit pas pourquoi le mur s’est fissuré et ne certifie pas sa sécurité.")
    st.subheader("Test guidé : trois visites")
    st.button("Charger l’exemple fictif", icon=":material/science:", on_click=enter_demo)
    st.write("Dans Carnet de visites, sélectionne Pavillon Atelier puis F-01. Les dates sont espacées de 15 jours et se terminent aujourd’hui.")
    st.table(pd.DataFrame({"Visite": ["J−30", "J−15", "Aujourd’hui"], "Répétitions (mm)": ["0,29 ; 0,30 ; 0,31", "0,44 ; 0,45 ; 0,46", "0,59 ; 0,60 ; 0,61"], "Moyenne attendue (mm)": [.30, .45, .60]}), alt="Données fictives pour tester trois visites.")
    note("Résultats attendus pour F-01 : +0,30 mm en 30 jours ; +0,010 mm/j sur la période. Pour F-02 : variation globale nulle, malgré une mesure intermédiaire différente.")
    st.subheader("Tester chaque outil")
    st.markdown("""
1. **Observatoire** : l’exemple contient 1 bâtiment, 2 repères et 6 visites.
2. **Bâtiments et repères** : clique sur le mur, crée F-03 ; le repère apparaît après enregistrement. Ce clic ne mesure aucune largeur.
3. **Carnet de visites** : ajoute une visite à F-03 avec `0,29 ; 0,30 ; 0,31`, u = `0,02` mm. La moyenne vaut 0,30 mm et l’écart-type 0,01 mm.
4. **Modifier** : change les mesures de cette visite en `0,39 ; 0,40 ; 0,41`. La moyenne devient 0,40 mm, sans créer une seconde visite.
5. **Comparateur photo** : choisis F-01, première et dernière images. Le curseur révèle les illustrations fictives avant/après. La variation numérique vaut +0,30 mm.
6. **Mesures et incertitudes** : F-01, première et dernière visites, seuil 0,10 mm. u visite ≈ 0,02082 mm ; u différence ≈ 0,02944 mm ; intervalle approximatif [0,2423 ; 0,3577] mm. Le pourcentage du modèle est proche de 100 %, pas un risque structurel.
7. **Rapports** : ouvre PDF, CSV, Excel et JSON. Réimporte le JSON dans l’exemple : un nouveau bâtiment « copie » est ajouté, l’ancien reste intact.
8. **Erreurs** : une mesure négative ou non numérique est refusée. Ajouter deux visites à la même date pour le même repère est refusé ; choisis « Modifier ».
9. **Fiches fissures** : clique sur F-01 dans la vue annotée. Sa prochaine visite est prévue dans 7 jours. Retire la case de planification puis enregistre : le statut devient « Suivi à jour ».
10. **Nouvelle inspection** : crée « Maison test », passe sans photo, crée F-TEST et saisis `0,29 ; 0,30 ; 0,31`. Le résumé donne 0,300 mm. Le dossier est ajouté uniquement après « Enregistrer l’inspection ».
11. **Zoom** : dans le comparateur, agrandis à 2× puis déplace l’image. Les deux vues bougent ensemble. « Réinitialiser » revient à 1×.
""")
    st.subheader("Bonnes pratiques de relevé")
    st.write("Choisis un repère fixe, une unité constante et une méthode de mesure reproductible. Note le matériel, l’opérateur et les conditions utiles. L’instrument doit réellement permettre de lire les décimales saisies. Une photo différente ne remplace pas une mesure.")
    st.caption("Aucun classement automatique en « sûr » ou « dangereux ». Pour une situation réelle préoccupante, le carnet ne remplace pas une inspection par une personne qualifiée.")


def main():
    st.set_page_config(page_title="FissureScope | Carnet d’inspection", page_icon=":material/center_focus_strong:", layout="wide")
    THEME_BRIDGE(data=dict(mode="dark" if visual_palette()["dark"] else "light"), key="theme_bridge",
                 on_mode_change=lambda: None, width=1, height=0)
    apply_visual_style()
    st.session_state.setdefault("space", "Mes données")
    st.session_state.setdefault("demo", empty_data())
    st.session_state.setdefault("demo_revision", 0)
    st.session_state.setdefault("page", "Observatoire")
    if st.session_state.page not in PAGES + ["Nouvelle inspection"]:
        st.session_state.page = "Observatoire"
    with st.sidebar:
        st.html('<div class="side-logo">'+LOGO+'<span>FissureScope</span></div>')
        st.caption("ATELIER D’OBSERVATION")
        space = st.selectbox("Espace de travail", ["Mes données", "Exemple fictif"], key="space")
        st.button("Charger l’exemple fictif", icon=":material/science:", on_click=enter_demo, width="stretch")
        st.button("Nouvelle inspection", icon=":material/add_a_photo:", type="primary",
                  key="sidebar_inspect", on_click=start_inspection, width="stretch")
        st.space("small")
        for page, icon in zip(PAGES, ICONS):
            st.button(page, icon=f":material/{icon}:", key="nav_"+page, width="stretch",
                      type="primary" if st.session_state.page == page else "secondary", on_click=switch_page, args=(page,))
        st.space("medium")
        with st.expander("Apparence", icon=":material/contrast:"):
            st.caption("Clair ou sombre : ouvrir le menu ⋮ en haut à droite, puis choisir Light (clair), Dark (sombre) ou System (automatique). Le réglage est conservé par ce navigateur.")
        st.caption(f"Version {VERSION} · sans diagnostic")
    if space == "Exemple fictif":
        data, revision = deepcopy(st.session_state.demo), st.session_state.demo_revision
    else:
        try:
            data, revision = load_database()
        except (ValueError, sqlite3.Error, OSError, json.JSONDecodeError) as ex:
            st.error(f"Impossible d’ouvrir le registre : {ex}")
            st.stop()
    st.html('<div class="mast"><div><div class="brand">'+LOGO+'<span class="wordmark">Fissure<em>Scope</em></span></div><div class="subtitle">Carnet d’inspection des bâtiments</div></div>'
            f'<div class="stamp">{"EXEMPLE FICTIF" if space == "Exemple fictif" else "DONNÉES SAISIES"} / {today().strftime("%d.%m.%Y")}</div></div>')
    st.html('<div class="eyebrow">BÂTIMENTS / OBSERVATIONS / ÉVOLUTION</div>')
    st.title(st.session_state.page)
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    if space == "Exemple fictif":
        st.caption("Espace d’essai séparé : mesures et illustrations fictives. Aucun mélange avec tes données.")
    handlers = dict(zip(PAGES, [dashboard, cracks_page, buildings_page, visits_page, compare_page,
                               uncertainty_page, reports_page, teaching_page]))
    handlers["Nouvelle inspection"] = inspection_page
    handlers[st.session_state.page](data, revision)
    st.html('<div class="footer">FissureScope / suivi pédagogique · Mesures saisies, pas de diagnostic structurel.</div>')


if __name__ == "__main__":
    main()
