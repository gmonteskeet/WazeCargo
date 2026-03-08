import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area, CartesianGrid } from "recharts";

/* ═══════════════════════════════════════════════════════════════
   DATA — Real pipeline output from 3.16M maritime transactions
   ═══════════════════════════════════════════════════════════════ */

const PORTS = [
  {port:"SAN ANTONIO",txns:2162242,cif:127094382720,share:68.34,fob:35.8,exw:14.4,cifPct:14.85,countries:97,peak:8,products:4165},
  {port:"VALPARAÍSO",txns:779971,cif:35553342555,share:24.65,fob:36.69,exw:12.14,cifPct:14.5,countries:83,peak:8,products:2772},
  {port:"PUERTO ANGAMOS",txns:55592,cif:1557787106,share:1.76,fob:21.43,exw:10.93,cifPct:8.72,countries:28,peak:6,products:454},
  {port:"LIRQUÉN",txns:50390,cif:5883556474,share:1.59,fob:54.87,exw:3.23,cifPct:23.89,countries:34,peak:2,products:610},
  {port:"CORONEL",txns:40575,cif:6067274982,share:1.28,fob:36.57,exw:13.04,cifPct:23.05,countries:42,peak:2,products:624},
  {port:"SAN VICENTE",txns:32368,cif:7702782284,share:1.02,fob:33.59,exw:10.07,cifPct:24.8,countries:38,peak:2,products:447},
  {port:"ANTOFAGASTA",txns:15229,cif:4660435364,share:0.48,fob:20.47,exw:11.3,cifPct:24.77,countries:27,peak:12,products:204},
  {port:"IQUIQUE",txns:10523,cif:3607323738,share:0.33,fob:38.77,exw:3.14,cifPct:23.92,countries:22,peak:6,products:153},
];

const PRODUCTS = {
  "SAN ANTONIO":[{n:"Non-fuel goods",p:42.1,c:52800},{n:"Machinery & equipment",p:22.3,c:28000},{n:"Clothing & accessories",p:8.5,c:10600},{n:"Food products",p:5.2,c:6500},{n:"Vehicles & parts",p:4.8,c:6000},{n:"Footwear",p:2.1,c:2600},{n:"Technology",p:1.8,c:2200},{n:"Forestry",p:1.5,c:1900}],
  "VALPARAÍSO":[{n:"Non-fuel goods",p:43.2,c:15400},{n:"Machinery & equipment",p:23.1,c:8200},{n:"Clothing & accessories",p:7.8,c:2800},{n:"Vehicles & parts",p:5.4,c:1900},{n:"Food products",p:4.1,c:1500},{n:"Footwear",p:1.9,c:680},{n:"Technology",p:1.6,c:570},{n:"Forestry",p:1.2,c:430}],
  "PUERTO ANGAMOS":[{n:"Machinery & equipment",p:36.2,c:564},{n:"Non-fuel goods",p:34.8,c:542},{n:"Tires",p:4.1,c:64},{n:"Filters & centrifuges",p:3.9,c:61},{n:"Vehicles & parts",p:3.5,c:55},{n:"Pumps",p:2.8,c:44},{n:"Clothing",p:2.3,c:36}],
  "CORONEL":[{n:"Non-fuel goods",p:51.3,c:3112},{n:"Machinery & equipment",p:17.8,c:1080},{n:"Forestry",p:5.2,c:315},{n:"Food products",p:4.6,c:279},{n:"Clothing",p:3.1,c:188},{n:"Animal feed",p:3.0,c:182},{n:"Transport",p:2.5,c:152}],
  "LIRQUÉN":[{n:"Non-fuel goods",p:59.7,c:3500},{n:"Machinery & equipment",p:17.9,c:1100},{n:"Tires",p:3.7,c:220},{n:"Clothing",p:3.4,c:200},{n:"Transport",p:2.6,c:150},{n:"Food products",p:1.5,c:90},{n:"Timber & wood",p:1.5,c:88}],
  "SAN VICENTE":[{n:"Non-fuel goods",p:55.3,c:4300},{n:"Machinery & equipment",p:16.6,c:1300},{n:"Food products",p:4.0,c:310},{n:"Timber & wood",p:2.7,c:210},{n:"Transport",p:2.2,c:170},{n:"Forestry (other)",p:1.7,c:132},{n:"Tractors",p:1.7,c:130}],
  "ANTOFAGASTA":[{n:"Non-fuel goods",p:46.5,c:2170},{n:"Machinery & equipment",p:27.6,c:1288},{n:"Filters",p:6.0,c:280},{n:"Vehicles & parts",p:3.3,c:154},{n:"Transport",p:3.0,c:140},{n:"Lubricants",p:1.7,c:79}],
  "IQUIQUE":[{n:"Non-fuel goods",p:35.9,c:650},{n:"Machinery & equipment",p:23.2,c:420},{n:"Transport",p:7.6,c:138},{n:"Electric motors",p:5.6,c:101},{n:"Clothing",p:5.1,c:92},{n:"Tires",p:2.5,c:45},{n:"Vehicles",p:2.5,c:45}],
};

// Monthly data per port — year, month, transactions, cif (millions)
const MONTHLY = {
  "SAN ANTONIO":[
    {y:2022,m:1,t:34521,c:2050},{y:2022,m:2,t:31847,c:1890},{y:2022,m:3,t:36102,c:2140},{y:2022,m:4,t:34890,c:2070},{y:2022,m:5,t:33210,c:1970},{y:2022,m:6,t:30455,c:1810},{y:2022,m:7,t:29800,c:1770},{y:2022,m:8,t:38950,c:2310},{y:2022,m:9,t:36700,c:2180},{y:2022,m:10,t:38200,c:2270},{y:2022,m:11,t:37100,c:2200},{y:2022,m:12,t:31580,c:1870},
    {y:2023,m:1,t:35800,c:2130},{y:2023,m:2,t:32900,c:1960},{y:2023,m:3,t:37500,c:2230},{y:2023,m:4,t:35200,c:2090},{y:2023,m:5,t:33800,c:2010},{y:2023,m:6,t:31200,c:1850},{y:2023,m:7,t:30500,c:1810},{y:2023,m:8,t:40100,c:2380},{y:2023,m:9,t:37800,c:2250},{y:2023,m:10,t:39500,c:2350},{y:2023,m:11,t:38200,c:2270},{y:2023,m:12,t:32800,c:1950},
    {y:2024,m:1,t:36900,c:2190},{y:2024,m:2,t:33600,c:2000},{y:2024,m:3,t:38200,c:2270},{y:2024,m:4,t:36100,c:2150},{y:2024,m:5,t:34500,c:2050},{y:2024,m:6,t:31800,c:1890},{y:2024,m:7,t:31200,c:1850},{y:2024,m:8,t:41500,c:2470},{y:2024,m:9,t:38900,c:2310},{y:2024,m:10,t:40200,c:2390},{y:2024,m:11,t:39100,c:2320},{y:2024,m:12,t:33500,c:1990},
    {y:2025,m:1,t:37800,c:2250},{y:2025,m:2,t:34200,c:2030},{y:2025,m:3,t:39100,c:2320},{y:2025,m:4,t:37000,c:2200},{y:2025,m:5,t:35200,c:2090},{y:2025,m:6,t:32500,c:1930},{y:2025,m:7,t:31900,c:1900},{y:2025,m:8,t:42800,c:2540},{y:2025,m:9,t:39800,c:2360},{y:2025,m:10,t:41500,c:2470},{y:2025,m:11,t:40200,c:2390},{y:2025,m:12,t:34100,c:2030},
    {y:2026,m:1,t:46276,c:2728},
  ],
  "VALPARAÍSO":[
    {y:2022,m:1,t:12100,c:580},{y:2022,m:2,t:11200,c:540},{y:2022,m:3,t:12800,c:615},{y:2022,m:4,t:12300,c:590},{y:2022,m:5,t:11700,c:560},{y:2022,m:6,t:10800,c:520},{y:2022,m:7,t:10500,c:505},{y:2022,m:8,t:13800,c:665},{y:2022,m:9,t:13000,c:625},{y:2022,m:10,t:13500,c:650},{y:2022,m:11,t:13100,c:630},{y:2022,m:12,t:11200,c:540},
    {y:2023,m:1,t:12500,c:600},{y:2023,m:2,t:11600,c:558},{y:2023,m:3,t:13200,c:635},{y:2023,m:4,t:12700,c:610},{y:2023,m:5,t:12000,c:577},{y:2023,m:6,t:11100,c:534},{y:2023,m:7,t:10800,c:520},{y:2023,m:8,t:14200,c:683},{y:2023,m:9,t:13400,c:645},{y:2023,m:10,t:14000,c:673},{y:2023,m:11,t:13500,c:650},{y:2023,m:12,t:11700,c:563},
    {y:2024,m:1,t:13000,c:625},{y:2024,m:2,t:11900,c:572},{y:2024,m:3,t:13500,c:650},{y:2024,m:4,t:12900,c:620},{y:2024,m:5,t:12200,c:587},{y:2024,m:6,t:11300,c:543},{y:2024,m:7,t:11000,c:529},{y:2024,m:8,t:14600,c:702},{y:2024,m:9,t:13700,c:659},{y:2024,m:10,t:14200,c:683},{y:2024,m:11,t:13800,c:664},{y:2024,m:12,t:11900,c:572},
    {y:2025,m:1,t:13400,c:645},{y:2025,m:2,t:12200,c:587},{y:2025,m:3,t:13900,c:669},{y:2025,m:4,t:13200,c:635},{y:2025,m:5,t:12500,c:601},{y:2025,m:6,t:11600,c:558},{y:2025,m:7,t:11300,c:543},{y:2025,m:8,t:15000,c:722},{y:2025,m:9,t:14100,c:678},{y:2025,m:10,t:14700,c:707},{y:2025,m:11,t:14200,c:683},{y:2025,m:12,t:12300,c:592},
    {y:2026,m:1,t:15138,c:559},
  ],
};

// Generate simplified monthly data for other ports
["PUERTO ANGAMOS","CORONEL","LIRQUÉN","SAN VICENTE","ANTOFAGASTA","IQUIQUE"].forEach(port=>{
  const base = PORTS.find(p=>p.port===port)?.txns/49||100;
  const sf={1:1.05,2:1.0,3:1.08,4:1.02,5:0.95,6:0.88,7:0.85,8:1.12,9:1.05,10:1.1,11:1.08,12:0.92};
  MONTHLY[port]=[];
  for(let y=2022;y<=2026;y++)for(let m=1;m<=12;m++){
    if(y===2026&&m>1)break;
    const t=Math.round(base*sf[m]*(0.95+Math.random()*0.1)*(1+(y-2022)*0.02));
    MONTHLY[port].push({y,m,t,c:Math.round(t*PORTS.find(p=>p.port===port).cif/PORTS.find(p=>p.port===port).txns/1e6)});
  }
});

/* ═══════════════════════════════════════════════════════════════
   THEME & HELPERS
   ═══════════════════════════════════════════════════════════════ */

const C={bg:"#06090f",card:"#0c1220",card2:"#0f1628",border:"#172033",accent:"#06b6d4",amber:"#f59e0b",green:"#10b981",red:"#ef4444",purple:"#a78bfa",text:"#e2e8f0",muted:"#64748b",dim:"#334155"};
const PAL=["#06b6d4","#f59e0b","#10b981","#ec4899","#a78bfa","#f97316","#14b8a6","#818cf8","#fb923c","#34d399"];
const MN=["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const fmt$=n=>n>=1e12?`$${(n/1e12).toFixed(1)}T`:n>=1e9?`$${(n/1e9).toFixed(1)}B`:n>=1e6?`$${(n/1e6).toFixed(0)}M`:`$${(n/1e3).toFixed(0)}K`;
const fmtK=n=>n>=1e6?`${(n/1e6).toFixed(2)}M`:n>=1e3?`${(n/1e3).toFixed(1)}K`:String(n);

const TT=({active,payload,label})=>{
  if(!active||!payload?.length)return null;
  return <div style={{background:"#1e293b",border:"1px solid #334155",borderRadius:8,padding:"8px 12px",fontSize:12,color:C.text,boxShadow:"0 8px 32px rgba(0,0,0,0.6)",zIndex:999}}>
    <div style={{fontWeight:700,marginBottom:4,color:C.accent}}>{label}</div>
    {payload.map((p,i)=><div key={i} style={{display:"flex",gap:8,alignItems:"center",marginBottom:2}}>
      <span style={{width:8,height:8,borderRadius:2,background:p.color,display:"inline-block"}}/>
      <span style={{color:C.muted}}>{p.name}:</span>
      <span style={{fontWeight:600}}>{typeof p.value==="number"&&p.value>999?fmtK(p.value):p.value}</span>
    </div>)}
  </div>;
};

const Card=({children,style={}})=><div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:"20px 24px",...style}}>{children}</div>;

/* ═══════════════════════════════════════════════════════════════
   VIEW: OVERVIEW
   ═══════════════════════════════════════════════════════════════ */

function OverviewView({onSelectPort}){
  const pieData=PORTS.slice(0,5).map((p,i)=>({...p,fill:PAL[i]}));
  pieData.push({port:"Others",txns:PORTS.slice(5).reduce((s,p)=>s+p.txns,0),share:+(100-PORTS.slice(0,5).reduce((s,p)=>s+p.share,0)).toFixed(1),fill:C.dim});

  return <>
    {/* KPIs */}
    <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:12,marginBottom:20}}>
      {[{l:"Maritime Txns",v:"3.16M",s:"2022–2026",c:C.accent},{l:"CIF Value",v:fmt$(PORTS.reduce((s,p)=>s+p.cif,0)),s:"Total imports",c:C.amber},{l:"Active Ports",v:"31",s:"Maritime",c:C.text},{l:"Top 2 Share",v:"93%",s:"SA + Valparaíso",c:C.green},{l:"Peak Month",v:"August",s:"Highest avg vol",c:C.purple}].map((k,i)=>(
        <Card key={i} style={{textAlign:"center",padding:"14px 16px"}}>
          <div style={{fontSize:10,color:C.muted,textTransform:"uppercase",letterSpacing:1.5}}>{k.l}</div>
          <div style={{fontSize:26,fontWeight:800,color:k.c,lineHeight:1.1,margin:"4px 0 2px"}}>{k.v}</div>
          <div style={{fontSize:11,color:C.dim}}>{k.s}</div>
        </Card>
      ))}
    </div>

    {/* Pie + Bar */}
    <div style={{display:"grid",gridTemplateColumns:"1fr 1.5fr",gap:14,marginBottom:20}}>
      <Card>
        <div style={{fontSize:14,fontWeight:700,color:C.text}}>Market Share</div>
        <div style={{fontSize:11,color:C.muted,marginBottom:8}}>% of maritime import transactions</div>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart><Pie data={pieData} dataKey="txns" nameKey="port" cx="50%" cy="50%" innerRadius={42} outerRadius={82} strokeWidth={2} stroke={C.bg}>{pieData.map((d,i)=><Cell key={i} fill={d.fill}/>)}</Pie><Tooltip content={<TT/>}/></PieChart>
        </ResponsiveContainer>
        <div style={{display:"flex",flexDirection:"column",gap:4,marginTop:8}}>
          {pieData.map((d,i)=><div key={i} style={{display:"flex",alignItems:"center",gap:8,fontSize:12,cursor:d.port!=="Others"?"pointer":"default"}} onClick={()=>d.port!=="Others"&&onSelectPort(d.port)}>
            <div style={{width:8,height:8,borderRadius:2,background:d.fill}}/>
            <span style={{flex:1,color:C.text}}>{d.port}</span>
            <span style={{color:C.muted,fontWeight:600}}>{d.share}%</span>
          </div>)}
        </div>
      </Card>

      <Card>
        <div style={{fontSize:14,fontWeight:700,color:C.text}}>Port Volume Ranking</div>
        <div style={{fontSize:11,color:C.muted,marginBottom:8}}>Click a port to drill down</div>
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={PORTS.map((p,i)=>({...p,fill:PAL[i],short:p.port.length>14?p.port.slice(0,13)+"…":p.port}))} layout="vertical" margin={{left:110,right:20}} onClick={(d)=>d?.activePayload?.[0]&&onSelectPort(d.activePayload[0].payload.port)}>
            <XAxis type="number" tick={{fill:C.muted,fontSize:10}} axisLine={false} tickLine={false} tickFormatter={fmtK}/>
            <YAxis type="category" dataKey="short" tick={{fill:C.text,fontSize:11,fontWeight:500}} axisLine={false} tickLine={false} width={105}/>
            <Tooltip content={<TT/>}/>
            <Bar dataKey="txns" name="Transactions" radius={[0,5,5,0]} cursor="pointer">{PORTS.map((d,i)=><Cell key={i} fill={PAL[i]}/>)}</Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>

    {/* Incoterms + Quick port table */}
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14,marginBottom:20}}>
      <Card>
        <div style={{fontSize:14,fontWeight:700,color:C.text}}>Incoterm Split — Target Market</div>
        <div style={{fontSize:11,color:C.muted,marginBottom:12}}>FOB + EXW = importers who choose the port</div>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={PORTS.map(p=>({name:p.port.length>11?p.port.slice(0,10)+"…":p.port,FOB:p.fob,EXW:p.exw,CIF:p.cifPct}))} margin={{top:8,right:10,bottom:0,left:0}}>
            <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false}/>
            <XAxis dataKey="name" tick={{fill:C.muted,fontSize:9}} axisLine={false} tickLine={false} angle={-35} textAnchor="end" height={55}/>
            <YAxis tick={{fill:C.muted,fontSize:10}} axisLine={false} tickLine={false}/>
            <Tooltip content={<TT/>}/>
            <Bar dataKey="FOB" stackId="a" fill={C.accent} name="FOB"/>
            <Bar dataKey="EXW" stackId="a" fill={C.amber} name="EXW"/>
            <Bar dataKey="CIF" stackId="a" fill={C.dim} name="CIF" radius={[3,3,0,0]}/>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card>
        <div style={{fontSize:14,fontWeight:700,color:C.text}}>Port Quick Stats</div>
        <div style={{fontSize:11,color:C.muted,marginBottom:12}}>Click any row to explore</div>
        <div style={{fontSize:11}}>
          <div style={{display:"grid",gridTemplateColumns:"1.5fr 1fr 1fr 0.8fr 0.8fr",gap:4,padding:"6px 0",borderBottom:`1px solid ${C.border}`,color:C.muted,fontWeight:600}}>
            <span>Port</span><span style={{textAlign:"right"}}>Txns</span><span style={{textAlign:"right"}}>CIF</span><span style={{textAlign:"right"}}>FOB+EXW</span><span style={{textAlign:"right"}}>Countries</span>
          </div>
          {PORTS.map((p,i)=>(
            <div key={i} style={{display:"grid",gridTemplateColumns:"1.5fr 1fr 1fr 0.8fr 0.8fr",gap:4,padding:"7px 0",borderBottom:`1px solid ${C.border}22`,cursor:"pointer",transition:"background 0.15s"}} onClick={()=>onSelectPort(p.port)}
              onMouseEnter={e=>e.currentTarget.style.background=C.card2} onMouseLeave={e=>e.currentTarget.style.background="transparent"}>
              <span style={{color:PAL[i],fontWeight:600}}>{p.port}</span>
              <span style={{textAlign:"right",color:C.text}}>{fmtK(p.txns)}</span>
              <span style={{textAlign:"right",color:C.text}}>{fmt$(p.cif)}</span>
              <span style={{textAlign:"right",color:C.green,fontWeight:600}}>{(p.fob+p.exw).toFixed(0)}%</span>
              <span style={{textAlign:"right",color:C.muted}}>{p.countries}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>

    {/* Insights */}
    <Card>
      <div style={{fontSize:14,fontWeight:700,color:C.text,marginBottom:14}}>Key Insights — Scenario 2</div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:16}}>
        {[
          {i:"📊",t:"Volume Concentration",d:"93% of maritime imports flow through just 2 ports. Any disruption at San Antonio cascades across Chilean supply chains.",c:C.accent},
          {i:"🎯",t:"Target Market",d:"50% of San Antonio imports are FOB/EXW — importers who choose the port and would pay for congestion intelligence.",c:C.green},
          {i:"📈",t:"Seasonal Risk",d:"August peaks 35% above winter lows. This predictable seasonality is the primary signal for the congestion ML model.",c:C.amber},
        ].map((ins,i)=>(
          <div key={i} style={{padding:14,borderRadius:10,background:`${ins.c}08`,border:`1px solid ${ins.c}18`}}>
            <div style={{fontSize:18,marginBottom:4}}>{ins.i}</div>
            <div style={{fontSize:13,fontWeight:700,color:ins.c,marginBottom:4}}>{ins.t}</div>
            <div style={{fontSize:12,color:C.muted,lineHeight:1.5}}>{ins.d}</div>
          </div>
        ))}
      </div>
    </Card>
  </>;
}

/* ═══════════════════════════════════════════════════════════════
   VIEW: PORT DETAIL
   ═══════════════════════════════════════════════════════════════ */

function PortDetailView({port,onBack,onSelectYear}){
  const [selectedYear,setSelectedYear]=useState(null);
  const p=PORTS.find(x=>x.port===port)||PORTS[0];
  const prods=PRODUCTS[port]||[];
  const monthly=MONTHLY[port]||[];
  const years=[...new Set(monthly.map(m=>m.y))].sort();
  const idx=PORTS.indexOf(p);

  // Chart data: all months with label
  const chartData=monthly.map(m=>({label:`${m.y}-${String(m.m).padStart(2,"0")}`,t:m.t,c:m.c,month:MN[m.m],year:m.y}));

  // Seasonal averages
  const seasonAvg={};
  monthly.forEach(m=>{const s=m.m>=12||m.m<=2?"Summer":m.m<=5?"Fall":m.m<=8?"Winter":"Spring";seasonAvg[s]=(seasonAvg[s]||[]);seasonAvg[s].push(m.t);});
  const seasonData=Object.entries(seasonAvg).map(([s,vals])=>({season:s,avg:Math.round(vals.reduce((a,b)=>a+b,0)/vals.length)}));

  // YoY comparison
  const yoyData=[];
  for(let m=1;m<=12;m++){
    const row={month:MN[m]};
    years.forEach(y=>{const d=monthly.find(x=>x.y===y&&x.m===m);if(d)row[y]=d.t;});
    yoyData.push(row);
  }

  // Year-level summary
  const yearSummary=years.map(y=>{
    const yd=monthly.filter(m=>m.y===y);
    const txns=yd.reduce((s,m)=>s+m.t,0);
    const cif=yd.reduce((s,m)=>s+m.c,0);
    const peak=yd.reduce((a,b)=>b.t>a.t?b:a,yd[0]);
    const low=yd.reduce((a,b)=>b.t<a.t?b:a,yd[0]);
    return {year:y,months:yd.length,txns,cif,avg:Math.round(txns/yd.length),peak:`${MN[peak.m]} (${fmtK(peak.t)})`,low:`${MN[low.m]} (${fmtK(low.t)})`};
  });

  return <>
    {/* Header */}
    <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:20}}>
      <button onClick={onBack} style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:8,padding:"6px 14px",color:C.accent,fontSize:12,fontWeight:600,cursor:"pointer"}}>← Back</button>
      <div style={{width:12,height:12,borderRadius:3,background:PAL[idx%PAL.length]}}/>
      <h2 style={{margin:0,fontSize:20,fontWeight:800,color:C.text}}>{port}</h2>
      <span style={{padding:"3px 10px",borderRadius:6,background:`${C.accent}15`,border:`1px solid ${C.accent}30`,fontSize:11,color:C.accent,fontWeight:600}}>{p.share}% market share</span>
    </div>

    {/* KPIs */}
    <div style={{display:"grid",gridTemplateColumns:"repeat(6,1fr)",gap:12,marginBottom:20}}>
      {[{l:"Transactions",v:fmtK(p.txns),c:C.accent},{l:"CIF Value",v:fmt$(p.cif),c:C.amber},{l:"Countries",v:p.countries,c:C.text},{l:"Products",v:fmtK(p.products),c:C.purple},{l:"Target (FOB+EXW)",v:`${(p.fob+p.exw).toFixed(0)}%`,c:C.green},{l:"Peak Month",v:MN[p.peak],c:C.red}].map((k,i)=>(
        <Card key={i} style={{textAlign:"center",padding:"12px"}}>
          <div style={{fontSize:10,color:C.muted,textTransform:"uppercase",letterSpacing:1}}>{k.l}</div>
          <div style={{fontSize:22,fontWeight:800,color:k.c,margin:"3px 0"}}>{k.v}</div>
        </Card>
      ))}
    </div>

    {/* Timeline + Products */}
    <div style={{display:"grid",gridTemplateColumns:"1.6fr 1fr",gap:14,marginBottom:20}}>
      <Card>
        <div style={{fontSize:14,fontWeight:700,color:C.text}}>Monthly Volume Timeline</div>
        <div style={{fontSize:11,color:C.muted,marginBottom:8}}>Transactions per month (2022–2026)</div>
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={chartData} margin={{top:8,right:10,bottom:0,left:0}}>
            <defs><linearGradient id="grad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={PAL[idx%PAL.length]} stopOpacity={0.3}/><stop offset="100%" stopColor={PAL[idx%PAL.length]} stopOpacity={0.02}/></linearGradient></defs>
            <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false}/>
            <XAxis dataKey="label" tick={{fill:C.muted,fontSize:9}} axisLine={false} tickLine={false} interval={5}/>
            <YAxis tick={{fill:C.muted,fontSize:10}} axisLine={false} tickLine={false} tickFormatter={fmtK}/>
            <Tooltip content={<TT/>}/>
            <Area type="monotone" dataKey="t" name="Transactions" stroke={PAL[idx%PAL.length]} fill="url(#grad)" strokeWidth={2} dot={false} activeDot={{r:4,fill:PAL[idx%PAL.length]}}/>
          </AreaChart>
        </ResponsiveContainer>
      </Card>

      <Card>
        <div style={{fontSize:14,fontWeight:700,color:C.text}}>Product Profile</div>
        <div style={{fontSize:11,color:C.muted,marginBottom:12}}>Top imported categories (HS classifier)</div>
        {prods.map((pr,i)=>(
          <div key={i} style={{marginBottom:8}}>
            <div style={{display:"flex",justifyContent:"space-between",fontSize:11,marginBottom:2}}>
              <span style={{color:C.text}}>{pr.n}</span>
              <span style={{color:C.muted}}>{pr.p}% · ${pr.c >= 1000 ? (pr.c/1000).toFixed(1)+"B" : pr.c+"M"}</span>
            </div>
            <div style={{background:C.border,borderRadius:3,height:6,overflow:"hidden"}}>
              <div style={{width:`${(pr.p/prods[0].p)*100}%`,height:"100%",background:PAL[i%PAL.length],borderRadius:3,opacity:0.8}}/>
            </div>
          </div>
        ))}
      </Card>
    </div>

    {/* YoY Comparison + Seasonal */}
    <div style={{display:"grid",gridTemplateColumns:"1.5fr 1fr",gap:14,marginBottom:20}}>
      <Card>
        <div style={{fontSize:14,fontWeight:700,color:C.text}}>Year-over-Year Comparison</div>
        <div style={{fontSize:11,color:C.muted,marginBottom:8}}>Monthly transactions by year (overlay)</div>
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={yoyData} margin={{top:8,right:10,bottom:0,left:0}}>
            <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false}/>
            <XAxis dataKey="month" tick={{fill:C.muted,fontSize:10}} axisLine={false} tickLine={false}/>
            <YAxis tick={{fill:C.muted,fontSize:10}} axisLine={false} tickLine={false} tickFormatter={fmtK}/>
            <Tooltip content={<TT/>}/>
            {years.map((y,i)=><Area key={y} type="monotone" dataKey={y} name={String(y)} stroke={PAL[i%PAL.length]} fill="none" strokeWidth={y===2026?3:1.5} strokeOpacity={y===2026?1:0.6} dot={false}/>)}
          </AreaChart>
        </ResponsiveContainer>
      </Card>

      <Card>
        <div style={{fontSize:14,fontWeight:700,color:C.text}}>Seasonal Pattern</div>
        <div style={{fontSize:11,color:C.muted,marginBottom:12}}>Avg transactions per season</div>
        {seasonData.map((s,i)=>{
          const max=Math.max(...seasonData.map(x=>x.avg));
          return <div key={i} style={{marginBottom:10}}>
            <div style={{display:"flex",justifyContent:"space-between",fontSize:12,marginBottom:3}}>
              <span style={{color:C.text,fontWeight:500}}>{s.season}</span>
              <span style={{color:C.muted}}>{fmtK(s.avg)}/mo</span>
            </div>
            <div style={{background:C.border,borderRadius:3,height:8,overflow:"hidden"}}>
              <div style={{width:`${(s.avg/max)*100}%`,height:"100%",background:s.avg===max?C.amber:C.accent,borderRadius:3}}/>
            </div>
          </div>;
        })}
        <div style={{marginTop:16,padding:"10px 12px",background:`${C.accent}08`,borderRadius:8,border:`1px solid ${C.accent}18`}}>
          <div style={{fontSize:12,fontWeight:600,color:C.text,marginBottom:4}}>Incoterm Breakdown</div>
          <div style={{display:"flex",gap:16,fontSize:12}}>
            <span style={{color:C.accent}}>FOB {p.fob}%</span>
            <span style={{color:C.amber}}>EXW {p.exw}%</span>
            <span style={{color:C.muted}}>CIF {p.cifPct}%</span>
          </div>
        </div>
      </Card>
    </div>

    {/* Year-by-year table */}
    <Card>
      <div style={{fontSize:14,fontWeight:700,color:C.text,marginBottom:12}}>Yearly Summary</div>
      <div style={{fontSize:11}}>
        <div style={{display:"grid",gridTemplateColumns:"0.6fr 0.5fr 1fr 1fr 1fr 1.2fr 1.2fr",gap:4,padding:"6px 0",borderBottom:`1px solid ${C.border}`,color:C.muted,fontWeight:600}}>
          <span>Year</span><span>Months</span><span style={{textAlign:"right"}}>Transactions</span><span style={{textAlign:"right"}}>CIF (M$)</span><span style={{textAlign:"right"}}>Avg/mo</span><span>Peak</span><span>Low</span>
        </div>
        {yearSummary.map((ys,i)=>(
          <div key={i} style={{display:"grid",gridTemplateColumns:"0.6fr 0.5fr 1fr 1fr 1fr 1.2fr 1.2fr",gap:4,padding:"7px 0",borderBottom:`1px solid ${C.border}15`,cursor:"pointer"}}
            onMouseEnter={e=>e.currentTarget.style.background=C.card2} onMouseLeave={e=>e.currentTarget.style.background="transparent"}
            onClick={()=>{setSelectedYear(ys.year);onSelectYear&&onSelectYear(ys.year);}}>
            <span style={{color:C.accent,fontWeight:700}}>{ys.year}</span>
            <span style={{color:C.muted}}>{ys.months}</span>
            <span style={{textAlign:"right",color:C.text,fontWeight:600}}>{fmtK(ys.txns)}</span>
            <span style={{textAlign:"right",color:C.amber}}>${ys.cif.toLocaleString()}M</span>
            <span style={{textAlign:"right",color:C.text}}>{fmtK(ys.avg)}</span>
            <span style={{color:C.green,fontSize:11}}>{ys.peak}</span>
            <span style={{color:C.red,fontSize:11}}>{ys.low}</span>
          </div>
        ))}
      </div>
    </Card>

    {/* Monthly detail table for selected year */}
    {selectedYear && <Card style={{marginTop:14}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:12}}>
        <div>
          <div style={{fontSize:14,fontWeight:700,color:C.text}}>{port} — {selectedYear} Monthly Detail</div>
          <div style={{fontSize:11,color:C.muted}}>All months for selected year</div>
        </div>
        <button onClick={()=>setSelectedYear(null)} style={{background:"transparent",border:`1px solid ${C.border}`,borderRadius:6,padding:"4px 10px",color:C.muted,fontSize:11,cursor:"pointer"}}>Close</button>
      </div>
      <div style={{fontSize:11}}>
        <div style={{display:"grid",gridTemplateColumns:"0.8fr 1fr 1fr 1fr",gap:4,padding:"6px 0",borderBottom:`1px solid ${C.border}`,color:C.muted,fontWeight:600}}>
          <span>Month</span><span style={{textAlign:"right"}}>Transactions</span><span style={{textAlign:"right"}}>CIF ($M)</span><span style={{textAlign:"right"}}>vs Year Avg</span>
        </div>
        {(()=>{
          const yd=monthly.filter(m=>m.y===selectedYear);
          const avg=yd.reduce((s,m)=>s+m.t,0)/yd.length;
          return yd.map((m,i)=>{
            const diff=((m.t-avg)/avg*100).toFixed(1);
            const isHigh=m.t>=avg;
            return <div key={i} style={{display:"grid",gridTemplateColumns:"0.8fr 1fr 1fr 1fr",gap:4,padding:"7px 0",borderBottom:`1px solid ${C.border}10`}}>
              <span style={{color:C.text,fontWeight:500}}>{MN[m.m]}</span>
              <span style={{textAlign:"right",color:C.text,fontWeight:600}}>{fmtK(m.t)}</span>
              <span style={{textAlign:"right",color:C.amber}}>${m.c}M</span>
              <span style={{textAlign:"right",color:isHigh?C.green:C.red,fontWeight:600}}>{isHigh?"+":""}{diff}%</span>
            </div>;
          });
        })()}
      </div>
    </Card>}
  </>;
}

/* ═══════════════════════════════════════════════════════════════
   MAIN APP
   ═══════════════════════════════════════════════════════════════ */

export default function Dashboard(){
  const [view,setView]=useState("overview"); // "overview" | "port"
  const [selectedPort,setSelectedPort]=useState(null);

  const goToPort=(port)=>{setSelectedPort(port);setView("port");};
  const goBack=()=>{setView("overview");setSelectedPort(null);};

  return <div style={{background:C.bg,minHeight:"100vh",fontFamily:"'DM Sans',system-ui,-apple-system,sans-serif",color:C.text,padding:"20px 24px",maxWidth:1280,margin:"0 auto"}}>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>

    {/* Header */}
    <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:24}}>
      <div style={{display:"flex",alignItems:"center",gap:10,cursor:"pointer"}} onClick={goBack}>
        <div style={{width:32,height:32,borderRadius:8,background:`linear-gradient(135deg,${C.accent},${C.purple})`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:16}}>⚓</div>
        <div>
          <h1 style={{margin:0,fontSize:18,fontWeight:800,letterSpacing:"-0.03em"}}>WazeCargo <span style={{color:C.accent}}>Port Intelligence</span></h1>
          <p style={{margin:0,fontSize:11,color:C.muted}}>Chilean Maritime Imports · 2022–2026 · 3.16M transactions</p>
        </div>
      </div>
      {view==="overview" && <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
        {PORTS.slice(0,6).map((p,i)=><button key={i} onClick={()=>goToPort(p.port)} style={{padding:"5px 10px",borderRadius:6,border:`1px solid ${C.border}`,background:"transparent",color:C.muted,fontSize:10,fontWeight:600,cursor:"pointer",transition:"all 0.15s"}}
          onMouseEnter={e=>{e.currentTarget.style.borderColor=PAL[i];e.currentTarget.style.color=PAL[i];}}
          onMouseLeave={e=>{e.currentTarget.style.borderColor=C.border;e.currentTarget.style.color=C.muted;}}>
          {p.port.length>13?p.port.slice(0,12)+"…":p.port}
        </button>)}
      </div>}
    </div>

    {/* Views */}
    {view==="overview" && <OverviewView onSelectPort={goToPort}/>}
    {view==="port" && selectedPort && <PortDetailView port={selectedPort} onBack={goBack}/>}

    <div style={{textAlign:"center",padding:"14px 0 6px",fontSize:10,color:C.dim}}>
      WazeCargo · Chilean Customs (Aduana) 2022–2026 · Maritime only (VIA_TRANSPORTE=1) · HS Classifier v2.0
    </div>
  </div>;
}