/* ============ Two-pane — final direction (quiet editing) ============ */
function TwoPaneFinal({ dims, onChange, onDelete, onAdd, onReorder, discipline, onPreset }){
  const { SortableList, Grip, Ic, AutoTextarea, PresetDropdown } = window;
  const [sel, setSel] = React.useState(() => dims[0] && dims[0].id);

  React.useEffect(() => {
    if(!dims.find(d => d.id === sel)) setSel(dims[0] ? dims[0].id : null);
  }, [dims, sel]);

  const current = dims.find(d => d.id === sel);
  const curIndex = dims.findIndex(d => d.id === sel);
  const rowsRef = React.useRef(null);
  const wantScroll = React.useRef(false);

  React.useEffect(() => {
    if(wantScroll.current){
      wantScroll.current = false;
      const el = rowsRef.current;
      if(el) el.scrollTop = el.scrollHeight;
    }
  }, [dims.length]);

  const addAndReveal = () => {
    wantScroll.current = true;
    const id = onAdd();
    setSel(id);
  };

  return (
    <div className="tp2">
      <div className="tp2-left">
        <PresetDropdown value={discipline} onSelect={onPreset}/>
        <div className="tp2-rows" ref={rowsRef}>
          <SortableList items={dims} onReorder={onReorder} getKey={(d)=>d.id}>
          {(d, i, P) => (
            <div key={P.key}
              className={"tp-row" + (d.id === sel ? " sel" : "") + (P.state.dragging ? " drag-ghost" : "")}
              style={P.state.over ? {boxShadow:"inset 0 2px 0 var(--accent-2)"} : null}
              onClick={() => setSel(d.id)} {...P.rowProps}>
              <Grip {...P.handleProps}/>
              <span className="num">{i + 1}</span>
              <span className="tp-name">{d.name || <span style={{color:"var(--text-faint)"}}>Untitled</span>}</span>
              <button className="icon-btn danger tp-del" onClick={(e) => { e.stopPropagation(); onDelete(d.id); }} aria-label="Remove">
                <Ic.trash/>
              </button>
            </div>
          )}
          </SortableList>
        </div>
        <button className="add-row tp-add" onClick={addAndReveal}>
          <Ic.plus/> Add dimension
        </button>
      </div>

      {current ? (
        <div className="tp2-detail" key={current.id}>
          <div className="td2-top">
            <span className="num">{curIndex + 1}</span>
            <span className="td2-meta">of {dims.length}</span>
          </div>
          <input className="q-name" value={current.name}
            placeholder="Untitled dimension"
            onChange={(e) => onChange(current.id, "name", e.target.value)} autoFocus={!current.name}/>
          <AutoTextarea className="q-def" value={current.definition}
            placeholder="Add a definition — what should RegCheck look for on this dimension?"
            onChange={(e) => onChange(current.id, "definition", e.target.value)}/>
        </div>
      ) : (
        <div className="tp2-detail tp-empty">
          <Ic.empty style={{width:34,height:34,opacity:.5}}/>
          <div>Add a dimension to get started.</div>
        </div>
      )}
    </div>
  );
}
window.TwoPaneFinal = TwoPaneFinal;
