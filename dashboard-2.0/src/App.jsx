import { useState, useEffect, useMemo } from "react";
import { AreaChart, Area, ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Label, ResponsiveContainer } from "recharts";
import { MapContainer, TileLayer, CircleMarker, Tooltip as MapTooltip, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { COORDS, NAME2CODE, META, WSEAS, HIDDEN_PORTS } from "./refData";

/* ═══════════════════════════════════════════════════════════════════
   WazeCargo Port Intelligence — congestion x weather fused dashboard
   Data: 2026 forecasts (ml pipeline) + seasonal weather climatology
   (weather pipeline). All values from RDS, exported to /public/data.
   ═══════════════════════════════════════════════════════════════════ */

const C={bg:"#06090f",card:"#0c1220",card2:"#0f1628",border:"#172033",accent:"#06b6d4",amber:"#f59e0b",green:"#10b981",red:"#ef4444",purple:"#a78bfa",text:"#e2e8f0",muted:"#64748b",dim:"#334155",orange:"#f97316",swell:"#38bdf8",wind:"#a3e635"};
const PAL=["#06b6d4","#f59e0b","#10b981","#ec4899","#a78bfa","#f97316","#14b8a6","#818cf8","#fb923c","#34d399"];
const MN=["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const fmtK=n=>n>=1e6?`${(n/1e6).toFixed(1)}M`:n>=1e3?`${(n/1e3).toFixed(1)}K`:String(Math.round(n));

// congestion index color/label (0-1 scale)
function ciColor(ci){if(ci>=0.4)return C.red;if(ci>=0.25)return C.orange;if(ci>=0.15)return C.amber;return C.green;}
function ciLabel(ci){if(ci>=0.4)return"High";if(ci>=0.25)return"Elevated";if(ci>=0.15)return"Moderate";return"Low";}

// weather risk color/label (by % hours closed)
function wxColor(pct){if(pct>=8)return C.red;if(pct>=3)return C.orange;if(pct>=1)return C.amber;return C.green;}
function wxLabel(pct){if(pct>=8)return"High";if(pct>=3)return"Elevated";if(pct>=1)return"Moderate";return"Calm";}

// delay-risk label from the combined (weather-adjusted) congestion index
function delayLabel(ciAdj,pctClosed){
  if(ciAdj>=0.4||pctClosed>=8)return{t:"Critical",c:C.red};
  if(ciAdj>=0.25||pctClosed>=3)return{t:"High",c:C.orange};
  if(ciAdj>=0.15||pctClosed>=1)return{t:"Moderate",c:C.amber};
  return{t:"Low",c:C.green};
}

const TT=({active,payload,label})=>{
  if(!active||!payload?.length)return null;
  return <div style={{background:"#1e293b",border:"1px solid #334155",borderRadius:8,padding:"8px 12px",fontSize:12,color:C.text,boxShadow:"0 8px 32px rgba(0,0,0,0.6)"}}>
    <div style={{fontWeight:700,marginBottom:4,color:C.accent}}>{label}</div>
    {payload.map((p,i)=><div key={i} style={{display:"flex",gap:8,alignItems:"center",marginBottom:2}}>
      <span style={{width:8,height:8,borderRadius:2,background:p.color,display:"inline-block"}}/>
      <span style={{color:C.muted}}>{p.name}:</span>
      <span style={{fontWeight:600}}>{typeof p.value==="number"?(p.value<1&&p.value>0?(p.value*100).toFixed(0)+"%":p.value>999?fmtK(p.value):p.value):p.value}</span>
    </div>)}
  </div>;
};

function FlyTo({center,zoom}){const map=useMap();useEffect(()=>{map.flyTo(center,zoom,{duration:0.8});},[center,zoom,map]);return null;}

// weather code for an ML port name (via bridge); null if no weather coverage
const codeOf=name=>NAME2CODE[name]||null;
// NOTE: source weather values are stored as FRACTIONS (0-1). We surface them
// as percentages (0-100) everywhere in the UI, so scale at the read boundary.
function wxOf(name,month){
  const code=codeOf(name);
  if(!code||!WSEAS[code])return null;
  const w=WSEAS[code][month];
  if(!w)return null;
  // return a percent-scaled copy (closed/warn were fractions of hours)
  return {...w,
    closed:w.closed!=null?w.closed*100:null,
    warn:w.warn!=null?w.warn*100:null,
    closed_min:w.closed_min!=null?w.closed_min*100:null,
    closed_max:w.closed_max!=null?w.closed_max*100:null};
}
// annual avg % hours closed for a port (for the map ring) — returns percent
function wxAnnual(name){
  const code=codeOf(name);
  if(!code||!WSEAS[code])return null;
  const ms=Object.values(WSEAS[code]);
  if(!ms.length)return null;
  const meanFrac=ms.reduce((a,b)=>a+(b.closed||0),0)/ms.length;
  return meanFrac*100; // fraction -> percent
}

export default function Dashboard(){
  const [dir,setDir]=useState("import");
  const [selPort,setSelPort]=useState(null);
  const [selHS,setSelHS]=useState(null);
  const [mode,setMode]=useState("combined");  // congestion | weather | combined
  const [showExposed,setShowExposed]=useState(false);  // optional weather-exposed overlay (Combined mode)

  // data loaded from /public/data
  const [pf,setPf]=useState(null);      // port_forecast (raw + adjusted CI)
  const [pc,setPc]=useState(null);      // port_compare
  const [comm,setComm]=useState(null);  // commodity_forecast (heavy, lazy)
  const [loading,setLoading]=useState(true);
  const [commLoading,setCommLoading]=useState(false);

  // load all files on mount (commodity is heavy but user wants it ready)
  useEffect(()=>{
    Promise.all([
      fetch("/data/port_forecast.json").then(r=>r.json()),
      fetch("/data/port_compare.json").then(r=>r.json()),
    ]).then(([a,b])=>{setPf(a);setPc(b);setLoading(false);})
      .catch(e=>{console.error("data load failed",e);setLoading(false);});
    // commodity loads in parallel; dropdown fills when ready
    setCommLoading(true);
    fetch("/data/commodity_forecast.json").then(r=>r.json())
      .then(d=>{setComm(d);setCommLoading(false);})
      .catch(e=>{console.error("commodity load failed",e);setCommLoading(false);});
  },[]);

  /* ── derive everything from loaded data ─────────────────────────── */

  // unique ML ports with coords, excluding hidden junk categories
  const ports = useMemo(()=>{
    if(!pf)return [];
    const seen={};
    pf.forEach(r=>{ if(!HIDDEN_PORTS.includes(r.port)) seen[r.port]=true; });
    return Object.keys(seen).filter(p=>COORDS[p]);
  },[pf]);

  // per-port annual avg adjusted CI for current direction (for map + overview)
  const portCI = useMemo(()=>{
    if(!pf)return {};
    const acc={};
    pf.filter(r=>r.direction===dir&&!HIDDEN_PORTS.includes(r.port)).forEach(r=>{
      if(!acc[r.port])acc[r.port]={raw:0,adj:0,n:0,closed:0};
      acc[r.port].raw+=r.ci_raw; acc[r.port].adj+=r.ci_adjusted;
      acc[r.port].closed+=r.pct_hours_closed||0; acc[r.port].n++;
    });
    const out={};
    Object.entries(acc).forEach(([p,v])=>{
      // ci stays 0-1 (rendered with *100 at display); closed -> percent here
      out[p]={raw:v.raw/v.n, adj:v.adj/v.n, closed:(v.closed/v.n)*100};
    });
    return out;
  },[pf,dir]);

  // 12-month CI strips (raw + adjusted) for selected port
  const portMonthly = useMemo(()=>{
    if(!pf||!selPort)return null;
    const rows=pf.filter(r=>r.port===selPort&&r.direction===dir).sort((a,b)=>a.month-b.month);
    if(!rows.length)return null;
    return {
      raw:rows.map(r=>r.ci_raw),
      adj:rows.map(r=>r.ci_adjusted),
      ship:rows.map(r=>r.forecast_shipments),
      mult:rows.map(r=>r.weather_multiplier),
      type:rows.map(r=>r.adjustment_type),
      peak:rows.map(r=>r.at_or_above_hist_peak),
    };
  },[pf,selPort,dir]);

  // map port list sorted by adjusted CI
  const mapPorts = useMemo(()=>ports.map(name=>{
    const ci=portCI[name];
    const c=COORDS[name];
    return c&&ci?{name,ci:ci.adj,ciRaw:ci.raw,closed:ci.closed,lat:c[0],lng:c[1],wxAnnual:wxAnnual(name)}:null;
  }).filter(Boolean).sort((a,b)=>b.ci-a.ci),[ports,portCI]);

  // commodity list for current dir + selected port (from heavy file)
  const commByPort = useMemo(()=>{
    if(!comm)return {};
    const out={};
    comm.filter(r=>r.direction===dir).forEach(r=>{
      (out[r.port]=out[r.port]||[]).push(r);
    });
    return out;
  },[comm,dir]);

  // commodity 12-month strip for selPort + selHS
  const commFC = useMemo(()=>{
    if(!comm||!selPort||!selHS)return null;
    const rows=comm.filter(r=>r.port===selPort&&r.direction===dir&&r.hs2===selHS).sort((a,b)=>a.month-b.month);
    if(!rows.length)return null;
    const strip=Array(12).fill(0);
    rows.forEach(r=>{strip[r.month-1]=r.forecast_shipments;});
    return {strip,desc:rows[0].commodity_description,model:rows[0].model};
  },[comm,selPort,selHS,dir]);

  // cross-port comparison for selHS (weather-aware, from port_compare.json)
  const compareRows = useMemo(()=>{
    if(!pc||!selHS)return [];
    return pc.filter(r=>r.hs2===selHS&&r.direction===dir&&!HIDDEN_PORTS.includes(r.port))
             .sort((a,b)=>a.rank_in_commodity-b.rank_in_commodity);
  },[pc,selHS,dir]);

  // available commodities for the dropdown (port-scoped if a port is selected)
  const availComms = useMemo(()=>{
    if(!comm)return [];
    const pool = selPort&&commByPort[selPort] ? commByPort[selPort] : comm.filter(r=>r.direction===dir);
    const seen={};
    pool.forEach(r=>{ if(!seen[r.hs2]) seen[r.hs2]={hs:r.hs2,name:`HS${r.hs2} — ${(r.commodity_description||"").slice(0,40)}`}; });
    return Object.values(seen);
  },[comm,commByPort,selPort,dir]);

  const mapCenter = selPort&&COORDS[selPort]?COORDS[selPort]:[-33.5,-71.0];
  const mapZoom = selPort?7:4;

  const handleDir=d=>{setDir(d);setSelPort(null);setSelHS(null);};
  const handlePort=name=>{ if(selPort===name){setSelPort(null);setSelHS(null);} else setSelPort(name); };

  if(loading) return <div style={{background:C.bg,height:"100vh",display:"flex",alignItems:"center",justifyContent:"center",color:C.muted,fontFamily:"'DM Sans',system-ui"}}>Loading port intelligence…</div>;
  if(!pf) return <div style={{background:C.bg,height:"100vh",display:"flex",alignItems:"center",justifyContent:"center",color:C.red,fontFamily:"'DM Sans',system-ui"}}>Could not load /data/port_forecast.json — check the files are in public/data/</div>;

  /* ── layout ─────────────────────────────────────────────────────── */
  return (
    <div style={{background:C.bg,height:"100vh",fontFamily:"'DM Sans',system-ui,sans-serif",color:C.text,display:"flex",flexDirection:"column"}}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
      <style>{`
        .leaflet-tooltip{background:#1e293b!important;border:1px solid #334155!important;color:#e2e8f0!important;
          font-family:'DM Sans',system-ui,sans-serif!important;border-radius:8px!important;padding:6px 10px!important;
          box-shadow:0 4px 20px rgba(0,0,0,0.6)!important;font-size:12px!important}
        .leaflet-tooltip-top:before{border-top-color:#334155!important}
        select option{background:#0f1628;color:#e2e8f0}
        ::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0c1220}
        ::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}
      `}</style>

      {/* Header */}
      <div style={{padding:"12px 24px",borderBottom:`1px solid ${C.border}`,display:"flex",alignItems:"center",justifyContent:"space-between",flexShrink:0}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <div style={{width:32,height:32,borderRadius:8,background:`linear-gradient(135deg,${C.accent},${C.purple})`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:16}}>&#9875;</div>
          <div>
            <h1 style={{margin:0,fontSize:18,fontWeight:800,letterSpacing:"-0.03em"}}>WazeCargo <span style={{color:C.accent}}>Port Intelligence</span></h1>
            <p style={{margin:0,fontSize:10,color:C.muted}}>Congestion &times; Weather &middot; Delay-risk forecast 2026</p>
          </div>
        </div>
        <div style={{display:"flex",gap:14,fontSize:10,color:C.muted}}>
          <span><span style={{display:"inline-block",width:8,height:8,borderRadius:"50%",background:C.accent,marginRight:5}}/>Congestion</span>
          <span><span style={{display:"inline-block",width:8,height:8,borderRadius:"50%",border:`2px solid ${C.swell}`,marginRight:5}}/>Weather risk</span>
        </div>
      </div>

      {/* Filter bar */}
      <div style={{padding:"10px 24px",borderBottom:`1px solid ${C.border}`,display:"flex",alignItems:"center",gap:24,flexShrink:0,flexWrap:"wrap"}}>
        <div style={{display:"flex",gap:0}}>
          {["import","export"].map(d=>(
            <button key={d} onClick={()=>handleDir(d)}
              style={{padding:"6px 20px",fontSize:11,fontWeight:dir===d?700:500,cursor:"pointer",
                textTransform:"uppercase",letterSpacing:1,
                border:`1px solid ${dir===d?C.accent:C.border}`,
                borderRadius:d==="import"?"6px 0 0 6px":"0 6px 6px 0",
                background:dir===d?`${C.accent}20`:"transparent",
                color:dir===d?C.accent:C.muted}}>
              {d}
            </button>
          ))}
        </div>

        <div style={{display:"flex",alignItems:"center",gap:6}}>
          <span style={{fontSize:11,color:C.muted,fontWeight:600}}>Port:</span>
          {selPort
            ?<div style={{display:"flex",alignItems:"center",gap:6}}>
              <span style={{width:8,height:8,borderRadius:"50%",background:ciColor(portCI[selPort]?.adj??0)}}/>
              <span style={{fontSize:12,fontWeight:700,color:C.text}}>{selPort}</span>
              <button onClick={()=>{setSelPort(null);setSelHS(null);}}
                style={{width:18,height:18,borderRadius:4,border:`1px solid ${C.border}`,background:"transparent",
                  color:C.muted,cursor:"pointer",fontSize:12,lineHeight:1,display:"flex",alignItems:"center",justifyContent:"center"}}>×</button>
            </div>
            :<span style={{fontSize:11,color:C.dim,fontStyle:"italic"}}>Click map to select</span>
          }
        </div>

        <div style={{display:"flex",alignItems:"center",gap:6}}>
          <span style={{fontSize:11,color:C.muted,fontWeight:600}}>Commodity:</span>
          <select value={selHS||""} onChange={e=>setSelHS(e.target.value||null)}
            style={{padding:"5px 10px",paddingRight:28,borderRadius:6,border:`1px solid ${C.border}`,
              background:C.card2,color:C.text,fontSize:11,minWidth:240,cursor:"pointer",outline:"none",appearance:"auto"}}>
            <option value="">{comm?"— All commodities —":(commLoading?"loading…":"— select to load —")}</option>
            {availComms.map(c=><option key={c.hs} value={c.hs}>{c.name}</option>)}
          </select>
        </div>

        {/* Layer mode toggle */}
        <div style={{display:"flex",alignItems:"center",gap:8,marginLeft:"auto"}}>
          <span style={{fontSize:11,color:C.muted,fontWeight:600}}>View:</span>
          <div style={{display:"flex"}}>
            {[{k:"congestion",l:"Congestion"},{k:"weather",l:"Weather"},{k:"combined",l:"Combined"}].map((m,i)=>(
              <button key={m.k} onClick={()=>setMode(m.k)}
                style={{padding:"6px 14px",fontSize:11,fontWeight:mode===m.k?700:500,cursor:"pointer",
                  border:`1px solid ${mode===m.k?C.accent:C.border}`,
                  borderRadius:i===0?"6px 0 0 6px":i===2?"0 6px 6px 0":"0",
                  borderLeft:i>0?"none":undefined,
                  background:mode===m.k?`${C.accent}20`:"transparent",
                  color:mode===m.k?C.accent:C.muted}}>
                {m.l}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main: Map + Panel */}
      <div style={{display:"grid",gridTemplateColumns:"1.4fr 1fr",flex:1,overflow:"hidden"}}>
        {/* Map */}
        <div style={{position:"relative",borderRight:`1px solid ${C.border}`}}>
          <MapContainer center={[-33.5,-71.0]} zoom={4} style={{height:"100%",width:"100%"}} scrollWheelZoom={true} zoomControl={true}>
            <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution='&copy; <a href="https://carto.com/">CARTO</a>'/>
            <FlyTo center={mapCenter} zoom={mapZoom}/>
            {mapPorts.map(p=>{
              // mode-aware coloring. weather value = annual % hrs closed
              const wxC = p.wxAnnual!=null?wxColor(p.wxAnnual):C.dim;
              const isWxExposed = showExposed && p.wxAnnual!=null && p.wxAnnual>=1;
              let fill, ring, ringW;
              if(mode==="congestion"){ fill=ciColor(p.ci); ring=ciColor(p.ci); ringW=1.2; }
              else if(mode==="weather"){ fill=p.wxAnnual!=null?wxC:C.dim; ring=p.wxAnnual!=null?wxC:C.dim; ringW=1.2; }
              else { fill=ciColor(p.ci); ring=isWxExposed?C.swell:ciColor(p.ci); ringW=isWxExposed?2.5:1.2; }
              const sel=selPort===p.name;
              const rad = mode==="weather"
                ? (sel?14:(p.wxAnnual||0)>=8?10:(p.wxAnnual||0)>=3?7:5)
                : (sel?14:p.ci>0.3?10:p.ci>0.15?7:5);
              return (
              <CircleMarker key={p.name} center={[p.lat,p.lng]}
                radius={rad}
                pathOptions={{
                  color:sel?"#fff":ring,
                  fillColor:fill,fillOpacity:sel?0.9:0.65,
                  weight:sel?3:ringW}}
                eventHandlers={{click:()=>handlePort(p.name)}}>
                <MapTooltip direction="top" offset={[0,-8]} opacity={0.95}>
                  <div style={{fontWeight:700}}>{p.name}</div>
                  {mode!=="weather"&&<div style={{fontSize:11}}>Congestion: {(p.ci*100).toFixed(0)}% &middot; {ciLabel(p.ci)}</div>}
                  {mode!=="congestion"&&p.wxAnnual!=null&&<div style={{fontSize:11,color:"#38bdf8"}}>Weather closed: {p.wxAnnual.toFixed(1)}% hrs/yr &middot; {wxLabel(p.wxAnnual)}</div>}
                  {mode!=="congestion"&&p.wxAnnual==null&&<div style={{fontSize:11,color:C.dim}}>No weather coverage</div>}
                </MapTooltip>
              </CircleMarker>
            );})}
          </MapContainer>

          {/* Legend */}
          <div style={{position:"absolute",bottom:24,left:16,zIndex:1000,background:"rgba(12,18,32,0.92)",borderRadius:10,padding:"10px 14px",border:`1px solid ${C.border}`,fontSize:10}}>
            <div style={{fontWeight:700,color:C.text,marginBottom:6,textTransform:"uppercase",letterSpacing:1}}>
              {dir} {mode==="congestion"?"congestion":mode==="weather"?"weather risk":"delay risk"}
            </div>
            {(mode==="weather"
              ?[{l:"Calm (<1%)",c:C.green},{l:"Moderate (1-3%)",c:C.amber},{l:"Elevated (3-8%)",c:C.orange},{l:"High (>8%)",c:C.red}]
              :[{l:"Low (<15%)",c:C.green},{l:"Moderate (15-25%)",c:C.amber},{l:"Elevated (25-40%)",c:C.orange},{l:"High (>40%)",c:C.red}]
            ).map(x=>(
              <div key={x.l} style={{display:"flex",alignItems:"center",gap:6,marginBottom:3}}>
                <span style={{width:8,height:8,borderRadius:"50%",background:x.c,display:"inline-block"}}/>
                <span style={{color:C.muted}}>{x.l}</span>
              </div>
            ))}
            {mode==="combined"&&<div onClick={()=>setShowExposed(v=>!v)}
              style={{display:"flex",alignItems:"center",gap:6,marginTop:4,paddingTop:6,borderTop:`1px solid ${C.border}`,cursor:"pointer",userSelect:"none"}}>
              <span style={{width:14,height:14,borderRadius:4,border:`1px solid ${showExposed?C.swell:C.dim}`,
                background:showExposed?C.swell:"transparent",display:"flex",alignItems:"center",justifyContent:"center",
                fontSize:10,color:"#000",fontWeight:800}}>{showExposed?"✓":""}</span>
              <span style={{width:8,height:8,borderRadius:"50%",border:`2px solid ${C.swell}`,display:"inline-block"}}/>
              <span style={{color:showExposed?C.text:C.muted}}>Highlight weather-exposed (≥1%)</span>
            </div>}
          </div>
        </div>

        {/* Detail panel */}
        <div style={{overflowY:"auto",padding:"16px 20px",display:"flex",flexDirection:"column",gap:12}}>
          {!selPort&&!selHS&&<OverviewPanel ports={mapPorts} dir={dir} mode={mode} showExposed={showExposed} onSelect={handlePort}/>}
          {!selPort&&selHS&&<CommodityComparePanel rows={compareRows} selHS={selHS} dir={dir} onSelectPort={setSelPort} loading={commLoading&&!comm}/>}
          {selPort&&<PortPanel selPort={selPort} dir={dir} mode={mode} ci={portCI[selPort]} monthly={portMonthly}
            selHS={selHS} commFC={commFC} compareRows={compareRows} commByPort={commByPort}
            onSelectHS={setSelHS} onSelectPort={setSelPort} commLoading={commLoading&&!comm}/>}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   PANEL: Overview (no port, no commodity)
   ═══════════════════════════════════════════════════════════════════ */
function OverviewPanel({ports,dir,mode,showExposed,onSelect}){
  if(!ports.length)return <div style={{color:C.muted,fontSize:12}}>No port data.</div>;
  // metric per mode: congestion->ci, weather->wxAnnual, combined->ci (adjusted)
  const wx = mode==="weather";
  const val = p => wx ? (p.wxAnnual??-1) : p.ci;
  const col = p => wx ? (p.wxAnnual!=null?wxColor(p.wxAnnual):C.dim) : ciColor(p.ci);
  const lab = p => wx ? (p.wxAnnual!=null?wxLabel(p.wxAnnual):"No data") : ciLabel(p.ci);
  const fmt = p => wx ? (p.wxAnnual!=null?p.wxAnnual.toFixed(1)+"%":"—") : (p.ci*100).toFixed(0)+"%";
  const sorted = [...ports].sort((a,b)=>val(b)-val(a));
  const hi=sorted[0], lo=sorted[sorted.length-1];
  const title = mode==="congestion"?"Congestion":mode==="weather"?"Weather":"Delay-Risk";
  const sub = mode==="congestion"?"ports ranked by trade-only congestion"
            : mode==="weather"?"ports ranked by % hours weather-closed (typical year)"
            : "ports ranked by weather-adjusted congestion. Fill = congestion, ring = weather";
  return <>
    <div style={{fontSize:15,fontWeight:700,color:C.text}}>{dir==="import"?"Import":"Export"} {title} Overview</div>
    <div style={{fontSize:11,color:C.muted,marginBottom:2}}>
      {sorted.length} {sub}. Click any port to drill in.
    </div>

    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:4}}>
      {[{l:wx?"Most Exposed":"Highest Risk",v:hi.name,s:fmt(hi),c:C.red},
        {l:wx?"Calmest":"Lowest Risk",v:lo.name,s:fmt(lo),c:C.green}].map((k,i)=>(
        <div key={i} style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:10,padding:"10px 12px",textAlign:"center"}}>
          <div style={{fontSize:9,color:C.muted,textTransform:"uppercase",letterSpacing:1.5}}>{k.l}</div>
          <div style={{fontSize:13,fontWeight:800,color:k.c,margin:"2px 0"}}>{k.v}</div>
          <div style={{fontSize:10,color:C.dim}}>{k.s}</div>
        </div>
      ))}
    </div>

    {sorted.map((p,i)=>(
      <div key={p.name} onClick={()=>onSelect(p.name)}
        style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"9px 12px",
          borderRadius:8,background:C.card2,border:`1px solid ${C.border}`,cursor:"pointer"}}>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{fontSize:10,fontWeight:800,color:C.dim,width:20}}>{i+1}</span>
          <span style={{width:8,height:8,borderRadius:"50%",background:col(p),
            boxShadow:showExposed&&mode==="combined"&&p.wxAnnual!=null&&p.wxAnnual>=1?`0 0 0 2px ${C.swell}`:"none"}}/>
          <span style={{fontSize:12,fontWeight:600,color:C.text}}>{p.name}</span>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          {showExposed&&mode==="combined"&&p.wxAnnual!=null&&p.wxAnnual>=1&&<span style={{fontSize:9,color:C.swell}}>⚓ {p.wxAnnual.toFixed(0)}%</span>}
          <span style={{fontSize:13,fontWeight:700,color:col(p)}}>{fmt(p)}</span>
          <span style={{fontSize:10,color:col(p)}}>{lab(p)}</span>
        </div>
      </div>
    ))}
  </>;
}

/* ═══════════════════════════════════════════════════════════════════
   PANEL: Commodity comparison (commodity selected, no port)
   "Which port should I use for my X export?" — weather-aware ranking
   ═══════════════════════════════════════════════════════════════════ */
function CommodityComparePanel({rows,selHS,dir,onSelectPort,loading}){
  if(loading)return <div style={{color:C.muted,fontSize:12}}>Loading commodity comparison…</div>;
  if(!rows.length)return <div style={{color:C.muted,fontSize:12}}>No comparison data for HS{selHS} ({dir}).</div>;
  const rec=rows[0];
  const desc=rec.commodity_description||"";
  return <>
    <div style={{fontSize:15,fontWeight:700,color:C.text}}>HS{selHS} — {desc.slice(0,46)}</div>
    <div style={{fontSize:11,color:C.muted}}>
      Best port for {dir}, ranked by forecast volume &times; (1 − weather-adjusted congestion).
    </div>

    <div style={{padding:"10px 14px",borderRadius:8,background:`${C.accent}08`,border:`1px solid ${C.accent}20`,fontSize:11,color:C.muted,lineHeight:1.6}}>
      <span style={{color:C.accent,fontWeight:700}}>Recommendation: </span>
      Ship through <span style={{color:C.green,fontWeight:700}}>{rec.port}</span> — {fmtK(rec.total_forecast_shipments)} shipments forecast,
      weather-adjusted congestion {(rec.avg_ci_adjusted*100).toFixed(0)}%
      {rec.avg_pct_hours_closed>0&&<span> ({(rec.avg_pct_hours_closed*100).toFixed(1)}% hrs weather-closed)</span>}.
    </div>

    <div style={{fontSize:12,fontWeight:600,color:C.text}}>Port Ranking</div>
    {rows.slice(0,10).map((p,i)=>(
      <div key={p.port} onClick={()=>onSelectPort(p.port)}
        style={{padding:"10px 12px",borderRadius:8,cursor:"pointer",
          background:i===0?`${C.green}10`:C.card2,border:`1px solid ${i===0?`${C.green}40`:C.border}`}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:3}}>
          <div style={{display:"flex",alignItems:"center",gap:8}}>
            <span style={{fontSize:11,fontWeight:800,color:C.dim}}>#{p.rank_in_commodity}</span>
            <span style={{fontSize:12,fontWeight:700,color:PAL[i%PAL.length]}}>{p.port}</span>
          </div>
          {i===0&&<span style={{padding:"2px 8px",borderRadius:4,background:C.green,color:"#000",fontSize:9,fontWeight:700}}>BEST</span>}
        </div>
        <div style={{display:"flex",gap:12,fontSize:11}}>
          <span style={{color:C.muted}}>{fmtK(p.total_forecast_shipments)} ships</span>
          <span style={{color:ciColor(p.avg_ci_adjusted)}}>congestion {(p.avg_ci_adjusted*100).toFixed(0)}%</span>
          {p.avg_pct_hours_closed>0&&<span style={{color:C.swell}}>⚓ {(p.avg_pct_hours_closed*100).toFixed(1)}%</span>}
        </div>
      </div>
    ))}
  </>;
}

/* ═══════════════════════════════════════════════════════════════════
   PANEL: Port detail — the fused view (congestion + weather)
   ═══════════════════════════════════════════════════════════════════ */
function PortPanel({selPort,dir,mode,ci,monthly,selHS,commFC,compareRows,commByPort,onSelectHS,onSelectPort,commLoading}){
  if(!monthly)return <div style={{color:C.muted,fontSize:12}}>No 2026 forecast for {selPort} ({dir}).</div>;
  const code=codeOf(selPort);
  const hasWx=!!(code&&WSEAS[code]);
  const showCong = mode!=="weather";   // congestion + combined show congestion
  const showWx   = mode!=="congestion"; // weather + combined show weather

  // seasonal weather strip (12 months) for this port
  const wxStrip = MN.slice(1).map((_,i)=>{ const w=wxOf(selPort,i+1); return w?w.closed:null; });

  // annual closure-driver split (swell vs wind) averaged over months that have data
  const drivers = (()=>{
    if(!hasWx)return null;
    let sw=0,wd=0,n=0;
    Object.values(WSEAS[code]).forEach(m=>{ if(m.swell!=null){sw+=m.swell;wd+=m.wind;n++;} });
    return n?{swell:sw/n,wind:wd/n}:null;
  })();

  const peakM=monthly.adj.indexOf(Math.max(...monthly.adj));
  const lowM=monthly.adj.indexOf(Math.min(...monthly.adj));
  const wxPeakM=hasWx?wxStrip.reduce((best,v,i)=>(v!=null&&(wxStrip[best]==null||v>wxStrip[best]))?i:best,0):-1;

  const ciAdj=ci?.adj??0, ciRaw=ci?.raw??0, closed=ci?.closed??0;
  const dl=delayLabel(ciAdj,closed);

  // chart: raw vs adjusted CI over 12 months
  const ciChart = monthly.raw.map((v,i)=>({month:MN[i+1],raw:v,adjusted:monthly.adj[i]}));

  const portComms = (commByPort[selPort]||[]);
  const topComms = (()=>{
    const agg={};
    portComms.forEach(r=>{ agg[r.hs2]=agg[r.hs2]||{hs:r.hs2,name:r.commodity_description,total:0}; agg[r.hs2].total+=r.forecast_shipments; });
    return Object.values(agg).sort((a,b)=>b.total-a.total).slice(0,8);
  })();

  return <>
    {/* Header — the fused delay-risk verdict */}
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start"}}>
      <div>
        <div style={{fontSize:18,fontWeight:800,color:C.text}}>{selPort}</div>
        <div style={{fontSize:11,color:C.muted,textTransform:"capitalize"}}>{dir} · {hasWx?META[code]?.zone+" coast":"no weather coverage"}</div>
      </div>
      <div style={{textAlign:"right"}}>
        <div style={{fontSize:10,color:C.muted,textTransform:"uppercase",letterSpacing:1}}>{mode==="congestion"?"Congestion risk":mode==="weather"?"Weather risk":"Delay risk"}</div>
        {mode==="weather"
          ?<>
            <div style={{fontSize:26,fontWeight:800,color:hasWx?wxColor(closed):C.dim,lineHeight:1}}>{hasWx?wxLabel(closed):"No data"}</div>
            <div style={{fontSize:11,color:C.muted}}>{hasWx?<>{closed.toFixed(1)}% hrs closed/yr</>:"no weather coverage"}</div>
          </>
          :mode==="congestion"
          ?<>
            <div style={{fontSize:26,fontWeight:800,color:ciColor(ciRaw),lineHeight:1}}>{ciLabel(ciRaw)}</div>
            <div style={{fontSize:11,color:C.muted}}>congestion {(ciRaw*100).toFixed(0)}% (trade-only)</div>
          </>
          :<>
            <div style={{fontSize:26,fontWeight:800,color:dl.c,lineHeight:1}}>{dl.t}</div>
            <div style={{fontSize:11,color:C.muted}}>congestion {(ciAdj*100).toFixed(0)}%{hasWx&&<> · {closed.toFixed(1)}% hrs closed</>}</div>
          </>
        }
      </div>
    </div>

    {/* Plain-language explainer of what the numbers mean */}
    <div style={{fontSize:10,color:C.dim,padding:"8px 12px",borderRadius:8,background:C.card2,border:`1px solid ${C.border}`,lineHeight:1.5}}>
      {mode==="weather"
        ?<><strong style={{color:C.muted}}>Weather risk</strong> = share of hours the port is typically closed by swell/wind in an average year. Higher = more operational downtime.</>
        :mode==="congestion"
        ?<><strong style={{color:C.muted}}>Congestion index</strong> = forecast trade load vs this port's own historical range. 0% = quietest on record, 100% = at its busiest peak.</>
        :<><strong style={{color:C.muted}}>Delay risk</strong> combines congestion (trade load vs this port's historical range, 0–100%) with weather closures (% hours shut by swell/wind). Both push delays up.</>
      }
    </div>

    {/* Raw vs weather-adjusted congestion — the fusion, side by side */}
    {showCong&&<div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:"14px 16px"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:2}}>
        <div style={{fontSize:12,fontWeight:600,color:C.text}}>{mode==="congestion"?"Congestion forecast (trade-only)":"Congestion: trade-only vs weather-adjusted"}</div>
        <div style={{fontSize:10,color:C.muted}}>2026, monthly</div>
      </div>
      <div style={{fontSize:10,color:C.dim,marginBottom:8}}>
        {mode==="congestion"?"Expected port load from trade volume — higher = busier port.":"Dashed = trade demand alone; solid = after weather closures are factored in. Gap = weather impact."}
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <AreaChart data={ciChart} margin={{top:4,right:8,bottom:18,left:8}}>
          <defs>
            <linearGradient id="adjGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={C.accent} stopOpacity={0.3}/><stop offset="100%" stopColor={C.accent} stopOpacity={0.02}/>
            </linearGradient>
            <linearGradient id="rawGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={C.muted} stopOpacity={0.25}/><stop offset="100%" stopColor={C.muted} stopOpacity={0.02}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false}/>
          <XAxis dataKey="month" tick={{fill:C.muted,fontSize:10}} axisLine={false} tickLine={false}/>
          <YAxis tick={{fill:C.muted,fontSize:10}} axisLine={false} tickLine={false} tickFormatter={v=>(v*100).toFixed(0)+"%"} domain={[0,"auto"]} width={44}>
            <Label value="Congestion index" angle={-90} position="insideLeft" style={{fill:C.accent,fontSize:10,fontWeight:600,textAnchor:"middle"}}/>
          </YAxis>
          <Tooltip content={<TT/>}/>
          {mode==="congestion"
            ?<Area type="monotone" dataKey="raw" name="Congestion" stroke={C.accent} fill="url(#adjGrad)" strokeWidth={2.5} dot={{r:2,fill:C.accent}}/>
            :<>
              <Area type="monotone" dataKey="raw" name="Trade-only" stroke={C.muted} fill="none" strokeWidth={1.5} strokeDasharray="5 3" dot={false}/>
              <Area type="monotone" dataKey="adjusted" name="Weather-adjusted" stroke={C.accent} fill="url(#adjGrad)" strokeWidth={2.5} dot={{r:2,fill:C.accent}}/>
            </>
          }
        </AreaChart>
      </ResponsiveContainer>
      {mode!=="congestion"&&<div style={{display:"flex",gap:16,justifyContent:"center",marginTop:4,fontSize:10,color:C.muted}}>
        <span><span style={{display:"inline-block",width:12,height:0,borderTop:`2px dashed ${C.muted}`,marginRight:4,verticalAlign:"middle"}}/>Trade-only</span>
        <span><span style={{display:"inline-block",width:12,height:2,background:C.accent,marginRight:4,verticalAlign:"middle"}}/>Weather-adjusted</span>
      </div>}
    </div>}

    {/* Seasonal weather risk strip */}
    {showWx&&hasWx?(
      <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:"12px 14px"}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:2}}>
          <div style={{fontSize:12,fontWeight:600,color:C.text}}>Seasonal weather risk</div>
          <div style={{fontSize:10,color:C.muted}}>% hours port-closed, typical year</div>
        </div>
        <div style={{fontSize:10,color:C.dim,marginBottom:8}}>Each cell = a month. Number = % of hours the port is usually shut by weather. Plan around the red months.</div>
        <div style={{display:"grid",gridTemplateColumns:"repeat(12,1fr)",gap:3}}>
          {wxStrip.map((v,i)=>(
            <div key={i} style={{textAlign:"center",padding:"6px 2px",borderRadius:6,
              background:v==null?`${C.dim}15`:`${wxColor(v)}15`,border:`1px solid ${v==null?C.dim:wxColor(v)}25`}}>
              <div style={{fontSize:9,color:C.muted}}>{MN[i+1]}</div>
              <div style={{fontSize:11,fontWeight:700,color:v==null?C.dim:wxColor(v)}}>{v==null?"—":v.toFixed(1)}</div>
            </div>
          ))}
        </div>
        {drivers&&(
          <div style={{display:"flex",gap:8,marginTop:10,alignItems:"center"}}>
            <span style={{fontSize:10,color:C.muted}}>Closure driver:</span>
            <div style={{flex:1,height:8,borderRadius:4,overflow:"hidden",display:"flex"}}>
              <div style={{width:`${drivers.swell*100}%`,background:C.swell}}/>
              <div style={{width:`${drivers.wind*100}%`,background:C.wind}}/>
              <div style={{flex:1,background:C.dim}}/>
            </div>
            <span style={{fontSize:10,color:C.swell}}>swell {(drivers.swell*100).toFixed(0)}%</span>
            <span style={{fontSize:10,color:C.wind}}>wind {(drivers.wind*100).toFixed(0)}%</span>
          </div>
        )}
        {wxPeakM>=0&&<div style={{fontSize:11,color:C.muted,marginTop:8}}>
          Worst weather month: <strong style={{color:wxColor(wxStrip[wxPeakM])}}>{MN[wxPeakM+1]}</strong> ({wxStrip[wxPeakM].toFixed(1)}% hrs closed).
          {drivers&&drivers.swell>drivers.wind?" Swell-driven (marejadas).":" Wind-driven."}
        </div>}
      </div>
    ):showWx&&!hasWx?(
      <div style={{padding:"10px 14px",borderRadius:8,background:`${C.amber}08`,border:`1px solid ${C.amber}20`,fontSize:11,color:C.amber}}>
        No weather coverage for {selPort}{mode==="combined"?" — delay risk reflects congestion only.":"."}
      </div>
    ):null}

    {/* Best/worst months summary */}
    {showCong&&<div style={{display:"flex",gap:16,fontSize:11,padding:"8px 12px",borderRadius:8,background:C.card2,border:`1px solid ${C.border}`}}>
      <span style={{color:C.green}}>Best month: <strong>{MN[lowM+1]}</strong> ({(monthly.adj[lowM]*100).toFixed(0)}%)</span>
      <span style={{color:C.red}}>Worst: <strong>{MN[peakM+1]}</strong> ({(monthly.adj[peakM]*100).toFixed(0)}%)</span>
    </div>}

    {/* Commodity detail or list */}
    {selHS&&commFC?(
      <CommodityDetail selPort={selPort} selHS={selHS} commFC={commFC} monthly={monthly} mode={mode} compareRows={compareRows} onSelectPort={onSelectPort}/>
    ):commLoading?(
      <div style={{fontSize:12,color:C.muted}}>Loading commodities…</div>
    ):topComms.length>0?(
      <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:"12px 14px"}}>
        <div style={{fontSize:12,fontWeight:600,color:C.text,marginBottom:8}}>Top commodities — what drives the traffic</div>
        {topComms.map((c,i)=>(
          <div key={c.hs} onClick={()=>onSelectHS(c.hs)}
            style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"8px 10px",marginBottom:4,
              borderRadius:6,background:C.card2,border:`1px solid ${C.border}`,cursor:"pointer"}}>
            <span style={{fontSize:11,color:C.text}}>HS{c.hs} — {(c.name||"").slice(0,38)}</span>
            <span style={{fontSize:11,fontWeight:700,color:PAL[i%PAL.length]}}>{fmtK(c.total)}</span>
          </div>
        ))}
        <div style={{fontSize:10,color:C.dim,marginTop:4}}>Click a commodity for the cross-port view</div>
      </div>
    ):(
      <div style={{fontSize:12,color:C.muted,padding:12,background:C.card2,borderRadius:8}}>
        Select a commodity above to load the breakdown for {selPort}.
      </div>
    )}
  </>;
}

/* ── commodity detail inside a selected port ──────────────────────── */
function CommodityDetail({selPort,selHS,commFC,monthly,mode,compareRows,onSelectPort}){
  const total=commFC.strip.reduce((a,b)=>a+b,0);
  // congestion line: use weather-adjusted in weather/combined, trade-only in congestion mode
  const ciSeries = monthly ? (mode==="congestion"?monthly.raw:monthly.adj) : null;
  const data=commFC.strip.map((v,i)=>({
    month:MN[i+1],
    forecast:v,
    congestion: ciSeries?ciSeries[i]:null,
  }));
  const best=compareRows[0];
  const isBest=best&&best.port===selPort;
  const ciLabelTxt = mode==="congestion"?"Congestion (trade-only)":"Congestion (weather-adj.)";
  return <>
    <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:"14px 16px"}}>
      <div style={{fontSize:13,fontWeight:700,color:C.text}}>HS{selHS} — {(commFC.desc||"").slice(0,42)}</div>
      <div style={{fontSize:11,color:C.muted,marginBottom:8}}>2026 forecast: {fmtK(total)} shipments through {selPort}. Bars/area = volume, line = congestion.</div>
      <ResponsiveContainer width="100%" height={185}>
        <ComposedChart data={data} margin={{top:6,right:14,bottom:20,left:8}}>
          <defs><linearGradient id="cdGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={C.purple} stopOpacity={0.3}/><stop offset="100%" stopColor={C.purple} stopOpacity={0.02}/>
          </linearGradient></defs>
          <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false}/>
          <XAxis dataKey="month" tick={{fill:C.muted,fontSize:10}} axisLine={false} tickLine={false}/>
          {/* left axis = shipments */}
          <YAxis yAxisId="left" tick={{fill:C.muted,fontSize:10}} axisLine={false} tickLine={false} tickFormatter={fmtK} width={44}>
            <Label value="Shipments" angle={-90} position="insideLeft" style={{fill:C.purple,fontSize:10,fontWeight:600,textAnchor:"middle"}}/>
          </YAxis>
          {/* right axis = congestion % */}
          <YAxis yAxisId="right" orientation="right" domain={[0,"auto"]} tick={{fill:C.muted,fontSize:10}} axisLine={false} tickLine={false} tickFormatter={v=>(v*100).toFixed(0)+"%"} width={44}>
            <Label value="Congestion" angle={90} position="insideRight" style={{fill:C.accent,fontSize:10,fontWeight:600,textAnchor:"middle"}}/>
          </YAxis>
          <Tooltip content={<TT/>}/>
          <Area yAxisId="left" type="monotone" dataKey="forecast" name="Forecast (ships)" stroke={C.purple} fill="url(#cdGrad)" strokeWidth={2} dot={{r:2,fill:C.purple}}/>
          {ciSeries&&<Line yAxisId="right" type="monotone" dataKey="congestion" name={ciLabelTxt} stroke={C.accent} strokeWidth={2} strokeDasharray="5 3" dot={{r:2,fill:C.accent}}/>}
          <Legend wrapperStyle={{fontSize:10,paddingTop:6}} iconType="plainline"/>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
    {compareRows.length>1&&(
      <div style={{padding:"10px 14px",borderRadius:8,background:isBest?`${C.green}08`:`${C.amber}08`,border:`1px solid ${isBest?`${C.green}20`:`${C.amber}20`}`,fontSize:11,color:C.muted,lineHeight:1.6}}>
        <span style={{color:isBest?C.green:C.amber,fontWeight:700}}>{isBest?"This is the best port for this commodity.":"Better option available."}</span>
        {!isBest&&best&&<span> <span style={{color:C.green,fontWeight:600,cursor:"pointer",textDecoration:"underline"}} onClick={()=>onSelectPort(best.port)}>{best.port}</span> ranks #1 ({(best.avg_ci_adjusted*100).toFixed(0)}% congestion vs {selPort}).</span>}
      </div>
    )}
  </>;
}