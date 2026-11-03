"""Build static/places.json (place name -> states) and add district names to static/cities.json.
Source: the `indiapins` package (India Post pincode directory, data.gov.in open data). Run: pip install indiapins && python tools/build_places.py"""
import bz2, collections, json, os, re
import indiapins

STATES = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
          "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
          "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
          "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir",
          "Ladakh", "Lakshadweep", "Puducherry"]
LOOKUP = {re.sub(r"[^a-z]", "", s.lower()): s for s in STATES}
LOOKUP.update({"dadranagarhavelianddamandiu": STATES[30], "dadraandnagarhaveli": STATES[30], "damananddiu": STATES[30],
               "damananddiu": STATES[30], "orissa": "Odisha", "pondicherry": "Puducherry", "uttaranchal": "Uttarakhand"})


def norm_state(s):
    return LOOKUP.get(re.sub(r"[^a-z]", "", s.replace("&", "and").lower()))


text = bz2.open(os.path.join(os.path.dirname(indiapins.__file__), "pins.json.bz2")).read().decode("utf-8")
rows = [json.loads(line) for line in text.splitlines() if line.strip()]  # one JSON record per line
print("records:", len(rows))

places, districts = collections.defaultdict(set), collections.defaultdict(set)
for r in rows:
    st = norm_state((r.get("State") or ""))
    if not st:
        continue
    if st == "Jammu and Kashmir" and re.search(r"leh|kargil|ladakh", (r.get("District") or ""), re.I):
        st = "Ladakh"
    d = (r.get("District") or "").strip().title()
    if d:
        districts[st].add(d)
        places[re.sub(r"[^a-z ]", "", d.lower()).strip()].add(st)
    if r.get("BranchType") != "BO":  # towns have sub/head offices; skip village branch offices to keep the file small
        n = re.sub(r"\s*\b(s\.?o|h\.?o|g\.?p\.?o)\b\.?\s*$", "", (r.get("Name") or ""), flags=re.I)
        k = re.sub(r"[^a-z ]", "", n.lower()).strip()
        if len(k) > 2:
            places[k].add(st)

json.dump({k: sorted(v) for k, v in sorted(places.items())}, open("static/places.json", "w"), separators=(",", ":"))

cities = json.load(open("static/cities.json"))
for st, ds in districts.items():
    have = {c.lower() for c in cities.get(st, [])}
    cities[st] = sorted(set(cities.get(st, [])) | {d for d in ds if d.lower() not in have})
json.dump(cities, open("static/cities.json", "w"), ensure_ascii=False, indent=0)
print("places:", len(places), "| cities total:", sum(len(v) for v in cities.values()))
print("Baran ->", sorted(places.get("baran", [])), "| in Rajasthan list:", "Baran" in cities.get("Rajasthan", []))
