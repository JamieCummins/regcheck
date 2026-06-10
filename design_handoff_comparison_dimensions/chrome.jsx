/* ============ Shared chrome + primitives ============ */

/* ---- icons (simple line icons) ---- */
const Ic = {
  chevron: (p) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" {...p}><path d="M6 9l6 6 6-6"/></svg>,
  plus: (p) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" {...p}><path d="M12 5v14M5 12h14"/></svg>,
  x: (p) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" {...p}><path d="M6 6l12 12M18 6L6 18"/></svg>,
  trash: (p) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" {...p}><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>,
  check: (p) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" {...p}><path d="M4 12l5 5L20 6"/></svg>,
  info: (p) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></svg>,
  layers: (p) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}><path d="M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5"/></svg>,
  brain: (p) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" {...p}><path d="M9 5a3 3 0 0 0-3 3 3 3 0 0 0-1 5.5A2.5 2.5 0 0 0 7 18a2.5 2.5 0 0 0 2 1V5zM15 5a3 3 0 0 1 3 3 3 3 0 0 1 1 5.5A2.5 2.5 0 0 1 17 18a2.5 2.5 0 0 1-2 1V5z"/></svg>,
  pulse: (p) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>,
  trend: (p) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}><path d="M3 17l6-6 4 4 7-8M21 7v5h-5"/></svg>,
  globe: (p) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" {...p}><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.6 2.5 15.4 0 18M12 3c-2.5 2.6-2.5 15.4 0 18"/></svg>,
  empty: (p) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" {...p}><rect x="3" y="4" width="18" height="16" rx="2.5"/><path d="M3 9h18M8 14h8"/></svg>
};
const glyphFor = (g) => (Ic[g] || Ic.layers);

function Grip(props){
  return <span className="grip" {...props}><i/><i/><i/><i/><i/><i/></span>;
}

/* ---- auto-growing textarea ---- */
function AutoTextarea({ value, onChange, className, ...rest }){
  const ref = React.useRef(null);
  const fit = () => { const el = ref.current; if(el && el.offsetParent !== null){ el.style.height = "auto"; const extra = el.offsetHeight - el.clientHeight; el.style.height = (el.scrollHeight + extra) + "px"; } };
  React.useLayoutEffect(fit);              // refit on every render (covers becoming visible)
  React.useEffect(() => {
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, []);
  return <textarea ref={ref} className={className} value={value} rows={1}
    onChange={(e)=>{ onChange(e); fit(); }} {...rest} />;
}

/* ---- SortableList: drag the whole row, but only start from the handle ----
   children = (item, index, parts) => node
   parts.rowProps   -> spread on the draggable container
   parts.handleProps-> spread on the drag handle (Grip)
   parts.state      -> { dragging, over }
*/
function SortableList({ items, onReorder, getKey, children }){
  const [from, setFrom] = React.useState(null);
  const [over, setOver] = React.useState(null);
  const [armed, setArmed] = React.useState(null); // index allowed to drag (handle pressed)

  const reset = () => { setFrom(null); setOver(null); setArmed(null); };

  const handleProps = (i) => ({
    onMouseDown: () => setArmed(i),
    onMouseUp: () => setArmed(null),
    onTouchStart: () => setArmed(i),
  });
  const rowProps = (i) => ({
    draggable: armed === i,
    onDragStart: (e) => { setFrom(i); setOver(i); e.dataTransfer.effectAllowed = "move"; try{ e.dataTransfer.setData("text/plain", String(i)); }catch(_){} },
    onDragEnd: reset,
    onDragOver: (e) => { e.preventDefault(); if(from !== null && over !== i) setOver(i); },
    onDrop: (e) => { e.preventDefault(); if(from !== null && from !== i) onReorder(from, i); reset(); },
  });

  return items.map((item, i) =>
    children(item, i, {
      key: getKey ? getKey(item) : i,
      rowProps: rowProps(i),
      handleProps: handleProps(i),
      state: { dragging: from === i, over: over === i && from !== null && from !== i, from, over }
    })
  );
}

/* ---- Preset (discipline) dropdown ---- */
function PresetDropdown({ value, onSelect }){
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);
  const list = window.RC_DISCIPLINES;
  const cur = list.find(d => d.key === value) || list[0];
  React.useEffect(() => {
    const h = (e) => { if(ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);
  return (
    <div className="preset" ref={ref}>
      <button className={"preset-btn" + (open ? " open" : "")} onClick={() => setOpen(o => !o)}>
        <span className="pb-cap">Defaults</span>
        <span className="pb-cur"><span className="dot"></span>{cur.label}</span>
        <Ic.chevron className="chev"/>
      </button>
      {open && (
        <div className="preset-menu">
          <div className="pm-cap">Load defaults for…</div>
          {list.map(d => {
            const G = glyphFor(d.glyph);
            return (
              <button key={d.key} className={"preset-opt" + (d.key === value ? " on" : "")}
                onClick={() => { onSelect(d.key); setOpen(false); }}>
                <span className="po-ico"><G style={{width:16,height:16}}/></span>
                <span>
                  <div className="po-name">{d.label}</div>
                  <div className="po-meta">{d.meta}</div>
                </span>
                <Ic.check className="po-check"/>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ---- RegCheck app chrome ---- */
function RegNav(){
  return (
    <nav className="nav">
      <div className="brand"><span className="logo"></span>RegCheck</div>
      <div className="nav-links">
        <a href="#">HOME</a>
        <a href="#" className="on">TOOLS <Ic.chevron className="caret"/></a>
        <a href="#">TEAM</a>
        <a href="#">RESEARCH</a>
        <a href="#">CONTACT</a>
        <a href="#">FAQ</a>
      </div>
    </nav>
  );
}
function Progress(){
  return (
    <div className="progress">
      <div className="track"><span></span></div>
      <div className="step-label">STEP 4 OF 8</div>
    </div>
  );
}
function Footer(){
  return (
    <div className="foot">
      <button className="btn btn-ghost">Back</button>
      <button className="btn btn-primary">Next</button>
    </div>
  );
}

Object.assign(window, { Ic, glyphFor, Grip, AutoTextarea, SortableList, PresetDropdown, RegNav, Progress, Footer });
