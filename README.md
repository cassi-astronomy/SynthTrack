# SynthTrack 0.1 – kostra programu

GPU synthetic tracking pro hledání planetek a komet na sériích krátkých
expozic. Běží na **Windows i Linuxu**, ovládá se přes **webové GUI**
(lokálně i vzdáleně z jiného počítače) nebo z příkazové řádky.

> Stav: **funkční kostra**. Celý řetězec (kalibrace → umělý flat →
> registrace → odečet hvězd → synthetic tracking → prověření → identifikace
> → výstupy) běží a je otestovaný na syntetických datech. Na reálných
> snímcích se teprve bude ladit – viz „Co dál“.

## Instalace

Potřebujete Python 3.10 nebo novější ([python.org](https://www.python.org/downloads/), na Windows zaškrtnout „Add to PATH“).

```bash
cd SynthTrack
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux:  source .venv/bin/activate
pip install -e .
pip install -e ".[cuda12]"     # NVIDIA GPU (CUDA 12) – volitelné, jinak běží na CPU
pip install -e ".[cosmics]"    # volitelně odstranění kosmiky (astroscrappy)
```

Kontrola GPU: `synthtrack gpu` (má ukázat `"backend": "cupy"` a vaši kartu).

**ASTAP** (astrometrie, offline): nainstalujte ASTAP + databázi hvězd
(D50 pro běžná pole, D80 pro pole pod ~0,5°, W08 pro širokoúhlé objektivy)
z [hnsky.org/astap.htm](https://www.hnsky.org/astap.htm). Program ho najde
sám v `C:\Program Files\astap\` nebo v PATH, případně zadejte cestu v Nastavení.
Bez ASTAPu (a bez WCS v hlavičkách) program detekuje, ale neumí RA/Dec ani
identifikaci známých objektů.

## Spuštění

### Jako aplikace v okně (doporučeno na vlastním počítači)

```bash
pip install -e ".[desktop]"      # jednou: knihovna pywebview (okno přes Edge WebView2 / GTK)
synthtrack shortcut              # jednou: zástupce „SynthTrack“ na ploše a v nabídce Start
```

Pak se program spouští **dvojklikem na ikonu SynthTrack na ploše** – otevře se
samostatné okno bez prohlížeče a bez černé konzole. Výběr adresářů, import
katalogů a uložení exportu používají běžné systémové dialogy. Zavřením okna
se program ukončí (pokud běží úloha, zeptá se). Alternativy: dvojklik na
`SynthTrack.bat` ve složce programu, nebo příkaz `synthtrack app`.
Případné chyby se zapisují do `~/SynthTrackData/synthtrack-app.log`.

Na Linuxu potřebuje pywebview GTK nebo Qt (např. `sudo apt install python3-gi gir1.2-webkit2-4.1`
nebo `pip install pywebview[qt]`).

### V prohlížeči (vzdálený server, observatoř)

```bash
synthtrack serve                 # GUI na http://localhost:8765
synthtrack serve --host 0.0.0.0  # přístupné i z jiných počítačů v síti (Linux server na observatoři)
```

Windows: dvojklik na `start_synthtrack.bat`. Linux: `./start_synthtrack.sh`.

Datový adresář (nastavení, profily přístrojů, úlohy, katalogy) je
`~/SynthTrackData` – jiný lze zadat `--data CESTA` nebo proměnnou `SYNTHTRACK_DATA`.

### Rychlý test bez vlastních dat

```bash
synthtrack demo                                   # vytvoří syntetickou sekvenci se 3 planetkami a profil "demo"
synthtrack run --instrument demo --lights ~/SynthTrackData/demo/lights
```

## Jak to funguje

| Krok | Co se děje |
|---|---|
| Zorná pole | Adresář s více poli (např. 90 snímků z 5 polí mozaiky) se podle souřadnic v hlavičkách (WCS, jinak RA/DEC) rozdělí na samostatné úlohy, které se zpracují postupně. |
| Snímky | FITS 16 bit, čas středu expozice z DATE-OBS (+ EXPTIME/2), kontrola, že jde o jednu skupinu (expozice, filtr, binning). |
| Výběr masterů | Dark: shodná expozice (tolerance), binning, gain/offset, nejbližší teplota. Flat: stejný filtr (aliasy L/Lum/Clear…), binning, nejbližší datum. Volitelně škálování darku přes bias. Každá volba se zapíše i s důvodem. |
| Kalibrace | (light − dark) / normalizovaný flat, maska saturace, volitelně kosmika. |
| Umělý flat | Model pozadí ze sítě buněk se sigma-clippingem, potlačení hvězd mediánovým filtrem, spline interpolace; odečet a dělení mapou šumu → pixely v jednotkách σ. |
| Registrace | WCS všech snímků (pokud jsou), jinak fázová korelace se subpixelovou přesností. |
| Horké pixely | Defekty pevné vůči čipu se najdou díky driftu pole a zamaskují. |
| Odečet hvězd | Šablona pro každý snímek = medián jen ze snímků vzdálených v čase (pomalé planetky se tak samy neodečtou), maska jasných hvězd. |
| Matched filter | Gaussovský filtr s naměřeným FWHM, přenormování na jednotkovou σ. |
| Synthetic tracking | Mřížka rychlostí z rozsahu v ″/min, krok tak, aby chyba stopy < 0,7 px; pro každý pixel nejlepší SNR. Na GPU (CuPy) i CPU stejný kód. |
| Pomalé objekty | Zadané minimum rychlosti se vždy dodrží. Pokud je stopa proti FWHM velmi krátká, program upozorní na nižší účinnost, ale planetku z hledání svévolně nevyřadí. |
| Kandidáti | Lokální maxima nad prahem, jemné doladění rychlosti, sloučení duplicit. |
| Prověření | SNR ve 3 časových segmentech, poměr medián/průměr (jednosnímkové artefakty), reziduum hvězdy, pokrytí. |
| Identifikace | Lokální MPCORB + CometEls; rychlý dvoutělesový výběr kolem pole, pak **N-body integrace** z epochy elementů (Slunce, planety, Měsíc, Pluto, relativita), iterovaná doba letu světla, topocentrický pozorovatel; astrometrická poloha ICRF jako u MPC. Párování podle polohy a rychlosti, O−C. |
| Výstupy | Animace podstacků, průměrný a mediánový stack, seznam známých objektů v poli, export MPC 80 sloupců a CSV. |

## Profily přístrojů

V GUI „Přístroje“: ohnisko, pixel, binning, saturace, adresář s mastery,
tolerance, aliasy filtrů, vlastní FITS klíče a pravidla pro automatický režim
(např. `INSTRUME = ASI6200`, `FOCALLEN = 2000`). Měřítko a zorné pole se
dopočítá. Pro dlouhá ohniska zvažte softwarový binning 2.

Každý profil má vlastní **MPC kód a souřadnice stanoviště**. Je-li zapnuto
„Stanoviště z FITS“, zeměpisná šířka, délka a výška z hlavičky mají přednost;
MPC kód se vždy bere z profilu. Staré společné údaje v `settings.toml` se
použijí jen jako kompatibilní záloha, dokud nejsou profily doplněné.

Master snímky stačí mít v adresáři přístroje (i v podadresářích); typ se
pozná z IMAGETYP nebo z názvu souboru (dark/flat/bias).

## Více zorných polí v jednom adresáři

Program přečte pointaci každého snímku – střed obrazu z WCS v hlavičce, bez
WCS klíče `RA`/`DEC` (`OBJCTRA`/`OBJCTDEC`…, vlastní klíče v profilu) – a
zorné pole (z WCS, jinak z měřítka přístroje × rozměr snímku). Snímky, jejichž
pointace se liší méně než **tolerance × menší strana pole** (výchozí 0,25),
patří k jednomu poli; dithering a drift jsou mnohem menší, sousední pole
mozaiky jsou od sebe ~0,5–0,9 pole. Funguje i přes 0h RA a pro snímky
prokládané v čase (A B C D E A B C …).

* **GUI „Nová úloha“** – po výběru adresáře ukáže tabulku polí (RA, Dec,
  počet snímků, rozptyl ditheringu, čas); zaškrtnutá pole se založí jako
  samostatné úlohy „adresář · název pole (k/N)“ a zpracují se jedno po druhém.
  Název pole je `OBJECT`, pokud je pro pole jednoznačný, jinak „Pole k“.
* **Automatický režim** a **CLI** (`synthtrack run`) rozdělují stejně;
  `--field 2` zpracuje jen vybrané pole, `--no-split` vypne dělení.
* Pole s méně než 5 snímky a snímky bez souřadnic se vynechají (zapíše se do logu).
* Nastavení: „Katalogy a nastavení“ → *Zorná pole* (zapnuto/vypnuto, tolerance).
  Jsou-li pole velmi blízko (překryv > 75 %), toleranci snižte; při velkém
  ditheringu ji zvyšte – program varuje, když se rozptyl pole blíží toleranci.
* Pojistka: úloha, která přesto dostane adresář s více poli (stará úloha,
  REST API bez dělení), zpracuje jen největší pole a vypíše varování –
  pole se nikdy nesmíchají do jednoho stacku.

### Předem roztříděná pole v podadresářích

Vybraný adresář může místo snímků obsahovat podadresáře, každý s jedním polem
(i ve více úrovních, např. `noc/M31/…`, `noc/Pole2/…`):

```
noc_2026-10-23/
  Pole01/  *.fits
  Pole02/  *.fits
  …
```

* GUI ukáže tabulku podadresářů (počet snímků, expozice/filtr, střed pole,
  zda jsou k dispozici mastery); zaškrtnuté se založí jako úlohy
  „noc_2026-10-23/Pole01“ … a zpracují se postupně v pořadí názvů
  (Pole2 před Pole10). Název pole = název podadresáře.
* Podadresáře s darky/flaty/biasy (podle IMAGETYP), skryté adresáře a adresáře
  masterů přístrojů se přeskočí.
* Pokud některý podadresář přece jen obsahuje víc polí, rozdělí se ještě
  podle souřadnic (úlohy „…/B · Pole 1 (1/2)“).
* Totéž `synthtrack run --lights noc_2026-10-23` a automatický režim (ten
  bere každý podadresář jako samostatnou sadu).

## Automatický režim

Sleduje kořenový adresář včetně podadresářů. Jakmile adresář obsahuje
nastavený počet snímků a po dobu ustálení se nemění, spustí úlohu. Přístroj
se vybere pevně, nebo podle pravidel v hlavičce. Zpracované adresáře si
pamatuje (`watch_state.json`).

## Katalogy

„Katalogy a nastavení“ → *Aktualizovat z MPC* (nejdřív povolit stahování
z MPC – výchozí je úplně offline režim) nebo *Importovat ze souboru*
(MPCORB.DAT / .gz, CometEls.txt stažené jinde).

## Široké pole a pomalé planetky (ověřeno na reálných datech)

Testovací sada: CTA-S1 FRAM WF0, objektiv 135 mm + G4-16000, 14,1″/px, pole
16°×16°, 18 × 30 s s ditheringem, rozsah 57,6 min.

| Planetka | V | Posun za sekvenci | SNR | Pořadí (z 57 kandidátů) | Tycho-Tracker |
|---|---|---|---|---|---|
| (25) Phocaea | 13,8 | 2,8 px | 41,8 | 1. | ✓ |
| (983) Gunila | 15,7 | 2,8 px | 8,4 | 8. | ✗ |
| (1660) Wood | 15,4 | 6,5 px | 8,3 | 11. | ✓ |
| (1102) Pepita | 15,7 | 2,7 px | 6,3 | 45. | ✗ |

Proč běžný synthetic tracking pomalé objekty ztrácí a co dělá SynthTrack jinak:

* **Správná normalizace šumu.** Po odečtení šablony statického pole klesá
  u objektu, který se posune jen o ~1 FWHM, signál *i šum* součtu. Běžné
  √n podhodnocuje signifikanci až 2–3×. SynthTrack počítá šum součtu přesně
  (`search.TrackNorm`), šablona je σ-clipovaný průměr (statisticky optimální).
* **Šum měřený v čase** pro každý pixel + model šumu u hvězd (pozadí,
  Poisson hvězdy, residua úměrná jasu) – hvězdy nevyrábějí falešné detekce.
* **Omezení příspěvku jednoho snímku** (±4σ) – kosmika a horké pixely bez
  destruktivního maskování. Volitelné maskování kosmiky a družic je ve výchozím
  nastavení vypnuté, protože u podvzorkovaných hvězd může snižovat úplnost.
* **Fotometrické srovnání snímků** a odečet pomalých trendů pozadí (soumrak).
* **Prověření:** konzistence segmentů vůči *očekávanému* profilu pomalého
  objektu, χ² po snímcích, test opačného směru (−v), přesycené hvězdy.
* **Astrometrie:** fit PSF ve 3 podstackách → poloha, rychlost (±2–11 %),
  směr (±2°); do MPC exportu jdou změřené polohy.
* **Sesazení podle hvězd** s polynomem 3. řádu (distorze objektivu), RMS 0,06 px,
  maska horkých pixelů z ditheringu, FWHM odhadované po dlaždicích.
* Zpracování **po dlaždicích** přes soubory na disku – paměť nezávisí na velikosti čipu.

Doba zpracování 18 × 16 Mpx: ~4 min na 2jádrovém CPU bez GPU (kalibrace 39 s,
převzorkování 34 s, hledání 148 s, identifikace 13 s). Snímky i dlaždice se
zpracovávají paralelně ve vláknech (počet podle jader a volné RAM,
`cpu_workers` v nastavení), na více jádrech a s GPU úměrně rychleji.

## Přesnost poloh známých objektů

Ověřeno proti JPL Horizons (astrometrické RA/Dec, topocentricky) pro 5 planetek
z aktuálního MPCORB (epocha 9. 6. 2026), 4,5–7,5 měsíce od epochy:

| Objekt | N-body (DE421) | jen dvoutělesově |
|---|---|---|
| (1) Ceres | 0,03–0,04″ | 6–28″ |
| (4) Vesta | 0,03–0,04″ | 4–8″ |
| (10) Hygiea | 0,03–0,05″ | 13–88″ |
| (433) Eros (NEA) | 0,01–0,02″ | 2–5″ |
| (624) Hektor (trojan) | 0,02–0,03″ | 0,6–4″ |

Zbytek (setiny ″) je rozdíl mezi drahou MPC a drahou JPL. Integrátor sám drží
Merkur, Venuši a Mars proti JPL DE421 na < 5 km za rok; geometrie pozorování
sedí s astropy na tisíciny ″. 20 000 objektů se propaguje za ~3 s.

**Efemerida planet:** přibalená JPL DE421 2020–2035 (1,2 MB). Pro jiné roky
nebo nejvyšší přesnost vložte `de440s.bsp`
([NAIF](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/)) do
adresáře katalogů (`~/SynthTrackData/catalogs`). Bez souboru se mimo rozsah
použije ERFA (i tak < 0,1″).

Pozor: přesnost předpovědi je vždy omezená kvalitou dráhy – u planetek
pozorovaných v jedné opozici nebo u čerstvých objevů může být chyba MPCORB
i desítky ″, u komet navíc negravitační síly (zatím nemodelované).

## Pro vývojáře

Pravidla projektu, architektura, invarianty a referenční výsledky jsou
v [AGENTS.md](AGENTS.md) (určeno lidem i AI agentům, kteří na kódu pracují).

## Testy

```bash
pip install -e ".[dev]"
pytest
```

Testy ověřují: detekci syntetických planetek (drift, gradient, vinětace,
horké pixely), identifikaci známého objektu přes WCS + MPCORB, přesnost
poloh proti JPL Horizons (< 1″, reálně < 0,05″) a JPL DE421, geometrii proti
astropy, výběr masterů, formát MPC 80, automatický režim a REST API.

## Propustnost, injekce a klasifikátor (od 27. 10. 2026)

* **Mapa propustnosti** – u polí nízko nad obzorem, se zakrytým rohem (kopule, horizont) nebo s procházející oblačností se pro každý snímek změří z hvězd mapa propustnosti (mřížka 256 px). Šablona hvězdného pole se pak odečítá s touto mapou a místa s propustností pod 50 % se v daném snímku do hledání nepočítají. Při rovnoměrném poli se nic nemění. Nastavení: `throughput_maps`, `throughput_cell_px`, `throughput_min`.
* **Téměř statické zdroje** – detekce, jejíž změřený pohyb (fit PSF v podstackách) je pod 0,7× nejpomalejší hledané rychlosti, je reziduum hvězdy nebo proměnná hvězda a automaticky se zamítne.
* **Rezidua stejných hvězd bez ztráty detekcí** – filtr automaticky zamítnutých ve výchozím stavu seskupí opakované detekce u stejné statické hvězdy do jedné rozbalitelné položky. Všechny původní detekce zůstávají v `candidates.json`; seskupení nemění hledání, stav ani export a lze je vypnout. Slabý nový objekt u hvězdy proto není tiše odstraněn.
* **Injekce** – v Nové úloze lze zadat počet syntetických planetek (např. 50). Vloží se do snímků, projdou stejným hledáním a prověřením, v Souhrnu je úplnost podle SNR a nalezené jsou pod filtrem „injekce“ (v běžném seznamu se neukazují).
* **Klasifikátor** – tlačítko „Naučit klasifikátor“ v seznamu kandidátů se naučí z vašich rozhodnutí (Přijmout / Zamítnout) ve všech úlohách, ze známých planetek a z injekcí. Každý kandidát pak má skóre „ML“, lze podle něj řadit a skrýt nepravděpodobné (nic se nemaže). Označení se sčítají přes všechny úlohy; při smazání úlohy (i při každém učení) se uloží do `SynthTrackData/models/examples`, takže se neztratí. Model: logistická regrese na 15 příznacích; kvalita se křížově ověřuje po celých pozorovacích sériích, aby se kandidáti téhož pole nedostali současně do učení i testu. Neuronová síť dává smysl až se stovkami skutečných příkladů.
* **Spolehlivost** – každý kandidát má hodnocení vysoká / střední / nízká / žádná (podobně jako „Confidence“ v Tycho Trackeru). Vysoká: SNR ≥ 9, objekt ve všech třetinách sekvence, souhlasné snímky, astrometrie sedí, žádné varování; střední: SNR ≥ 7 a bez zásadních nesouladů; nízká: prošel prověřením, ale slabý; žádná: zamítnut prověřením. Najetím myši na štítek se ukáže, proč nemá vyšší. Lze podle ní řadit a filtrovat.
* **Orientace náhledů** – volba v liště: sever nahoře a východ vlevo (podle WCS, výchozí), jako FITS/DS9, nebo surově.
* **Porovnání s DSS** – tlačítko DSS u kandidáta (klávesa D) otevře okno: naše snímky jsou přes lokální WCS převzorkované na sever nahoru a východ vlevo, se stejným středem a měřítkem jako DSS2/2MASS. U starších úloh se lokální WCS dopočítá z okolních kandidátů. Zapíná se v Nastavení („Online: náhledy DSS“); rychlé zdroje STScI/NASA SkyView se zkoušejí před CDS hips2fits. Výřezy se ukládají k úloze.
* **Známé objekty** – kandidát, který prošel prověřením a je identifikován se známou planetkou, se automaticky označí jako přijatý.
* **Animace** – počet obrázků 2–6 nebo všechny snímky, úroveň černé/bílé (v násobcích šumu), gama a rychlost se nastavují v liště nad kandidáty a platí pro všechny; klávesy P (pauza), ←/→ (krok), X (zaměřovací kříž), D (DSS). Volby „dorovnat hvězdy“ (lokální korekce sesazení každého snímku) a „bez zakrytých“ (vynechá snímky s propustností pod 50 %). Lišta zůstává při posouvání nahoře.

## Co dál (plán)

1. **Ladění na reálných snímcích** různých přístrojů (135 mm … 2000 mm).
2. **Injekční testy** – (základ hotový) kalibrace prahů pro každý přístroj, MLP/CNN klasifikátor nad výřezy, až bude dost příkladů.
3. **Rychlost GPU** – fúzovaný CUDA kernel pro stackování, dlaždicové zpracování velkých čipů (60 Mpx), hierarchické prohledávání, více GPU.
4. **Přesná astrometrie** – refit WCS+SIP proti Gaia DR3 (vlastní pohyby), fotometrie v G, kontrola před odesláním na MPC, ADES export.
5. **Komety** – detekce difúzních objektů širším filtrem, negravitační parametry; poruchy od Ceres/Pallas/Vesty.
6. **Volitelné online služby** – SkyBoT/MPChecker, astrometry.net.
7. **ML klasifikátor** kandidátů učený na vašich rozhodnutích přijmout/zamítnout.

## Struktura

```
synthtrack/
  settings.py      nastavení (offline-first, GPU, ASTAP, parametry hledání, automatický režim)
  instruments.py   profily přístrojů, FITS klíče, aliasy filtrů
  fitsmeta.py      čtení FITS a metadat
  fields.py        rozdělení adresáře na zorná pole podle souřadnic, úloha na pole
  masters.py       knihovna masterů a automatický výběr dark/flat/bias
  calibration.py   kalibrace, umělý flat, normalizace šumu
  registration.py  detekce hvězd, párování, polynomiální distorze, převzorkování
  search.py        šablona, šum, matched filter, TrackNorm, hledání v dlaždici, prověření, astrometrie podstacků
  tracking.py      společné typy (Candidate, mřížka rychlostí), FWHM, deduplikace, záložní fázová korelace
  gpu.py           výběr backendu (CuPy/NumPy), stav GPU
  astrometry.py    ASTAP, WCS registrace
  mpcdata.py       MPCORB/CometEls – stažení, parsování, cache
  identify.py      předpověď pole, párování
  nbody.py         N-body integrátor, elementy → stav, doba letu světla
  ephem.py         efemeridy planet (JPL SPK / DE421 / ERFA)
  data/            JPL DE421 2020–2035
  pipeline.py      celý průběh jedné úlohy
  jobs.py          úlohy na disku + fronta
  watcher.py       automatický režim
  imaging.py       výřezy, animace
  export.py        MPC 80, CSV
  simulate.py      syntetická data pro testy a injekce
  web/             FastAPI + GUI (HTML/JS bez build kroku)
```
