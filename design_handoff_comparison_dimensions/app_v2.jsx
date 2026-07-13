/* ============ App v2 — two-pane, chosen direction ============ */
function AppV2(){
  const { RegNav, Progress, Footer, TwoPaneFinal } = window;

  const [discipline, setDiscipline] = React.useState("psychology");
  const [dims, setDims] = React.useState(() => window.RC_loadDiscipline("psychology"));

  const onChange = (id, field, val) =>
    setDims(ds => ds.map(d => d.id === id ? { ...d, [field]: val } : d));
  const onDelete = (id) =>
    setDims(ds => ds.filter(d => d.id !== id));
  const onAdd = () => {
    const nd = window.RC_newDim();
    setDims(ds => [...ds, nd]);
    return nd.id;
  };
  const onReorder = (from, to) =>
    setDims(ds => {
      const next = ds.slice();
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return next;
    });
  const onPreset = (key) => { setDiscipline(key); setDims(window.RC_loadDiscipline(key)); };

  return (
    <div className="app">
      <RegNav/>
      <Progress/>
      <div className="panel">
        <div className="panel-head" style={{marginBottom:24}}>
          <div className="ph-text">
            <h1>Comparison dimensions</h1>
            <p className="sub">RegCheck compares the registration and paper on each of these dimensions, in order.</p>
          </div>
        </div>
        <TwoPaneFinal dims={dims} onChange={onChange} onDelete={onDelete} onAdd={onAdd}
          onReorder={onReorder} discipline={discipline} onPreset={onPreset}/>
      </div>
      <Footer/>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<AppV2/>);
