import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area, ReferenceLine, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';
import { Activity, Heart, Zap, ArrowRight, AlertTriangle, CheckCircle2, User, ChevronLeft, Hexagon, Download } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';

// --- Sub-components (Inline for simplicity given file constraints) ---

const RangeInput = ({ label, value, min, max, onChange, unit, step = 1 }) => (
    <div className="mb-5">
        <div className="flex justify-between items-end mb-2">
            <label className="label-text mb-0 text-neutral-500">{label}</label>
            <div className="flex items-baseline gap-1">
                <input
                    type="number"
                    value={value}
                    onChange={(e) => onChange(Number(e.target.value))}
                    className="bg-transparent text-right text-teal-400 font-bold text-sm outline-none w-16 hover:text-teal-300 focus:text-white transition-colors"
                />
                <span className="text-neutral-600 text-xs">{unit}</span>
            </div>
        </div>
        <input
            type="range"
            min={min} max={max} step={step}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
        />
    </div>
);

const SelectInput = ({ label, value, options, onChange }) => (
    <div className="mb-5">
        <label className="label-text text-neutral-500">{label}</label>
        <div className="relative">
            <select
                value={value}
                onChange={(e) => onChange(Number(e.target.value))}
                className="w-full glass-input appearance-none cursor-pointer hover:border-teal-500/30 font-medium bg-neutral-900 border-neutral-800 text-neutral-300 focus:ring-teal-900"
            >
                {options.map(opt => (
                    <option key={opt.value} value={opt.value} className="bg-neutral-950 text-neutral-400">
                        {opt.label}
                    </option>
                ))}
            </select>
        </div>
    </div>
);

const StatCard = ({ label, value, change, unit, good }) => (
    <div className="glass-card p-5 flex flex-col justify-between h-full relative overflow-hidden group border border-white/5">
        <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity text-teal-500">
            <Activity className="w-8 h-8" />
        </div>
        <p className="text-neutral-500 text-xs font-bold uppercase tracking-wider">{label}</p>
        <div className="flex items-baseline gap-1 mt-2">
            <span className="text-2xl font-bold text-white tracking-tight">{value}</span>
            <span className="text-neutral-600 text-xs">{unit}</span>
        </div>
        {change && (
            <div className={`text-xs font-bold mt-2 flex items-center ${good ? 'text-emerald-400' : 'text-rose-400'}`}>
                {change > 0 ? '+' : ''}{change} <span className="ml-1 opacity-75 text-neutral-500 font-normal">projected</span>
            </div>
        )}
    </div>
);

// --- Main Dashboard ---

export default function Dashboard() {
    const navigate = useNavigate();

    // State
    const [formData, setFormData] = useState({
        Gender: 1.0, Age: 24, Race: 3, Education: 4,
        Calories: 1800, Protein: 80, Carbs: 250, Sugar: 40, Fiber: 25, Fat: 60,
        BMI: 21.6, SystolicBP: 110, DiastolicBP: 70, Glucose: 85, TotalCholesterol: 180,
        MedicationCount: 0, Diabetes: 0, Smoker: 0
    });

    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);

    const updateField = (field, val) => setFormData(prev => ({ ...prev, [field]: val }));

    const handlePredict = async () => {
        setLoading(true);
        try {
            // Predict Cluster (Returns everything now)
            const predRes = await axios.post('http://127.0.0.1:8000/predict_cluster', formData);

            setResults(predRes.data);
        } catch (e) {
            console.error(e);
            alert("Analysis failed. Check backend connection.");
        } finally {
            setLoading(false);
        }
    };

    const handleExportPDF = () => {
        try {
            if (!results) return;

            const doc = new jsPDF();

            // Brand Header
            doc.setFillColor(20, 184, 166); // Teal 500
            doc.rect(0, 0, 210, 20, 'F');
            doc.setFontSize(16);
            doc.setTextColor(255, 255, 255);
            doc.text("CORTEX MED // CLINICAL DOSSIER", 10, 13);

            // Metadata
            doc.setTextColor(0, 0, 0);
            doc.setFontSize(10);
            doc.text(`Generated: ${new Date().toLocaleString()}`, 10, 30);
            doc.text(`Patient ID: ${Math.random().toString(36).substr(2, 9).toUpperCase()}`, 150, 30);

            // Vitals Grid
            autoTable(doc, {
                startY: 35,
                head: [['Metric', 'Value', 'Metric', 'Value']],
                body: [
                    ['Age', formData.Age, 'Gender', formData.Gender === 1 ? 'Male' : 'Female'],
                    ['BMI', formData.BMI, 'Glucose', `${formData.Glucose} mg/dL`],
                    ['Systolic BP', `${formData.SystolicBP} mmHg`, 'Daily Calories', `${formData.Calories} kcal`]
                ],
                theme: 'grid',
                headStyles: { fillColor: [23, 23, 23] } // Neutral 900
            });

            // Classification
            doc.setFontSize(12);
            doc.setFont("helvetica", "bold");
            doc.text("Neural Segmentation Analysis", 10, doc.lastAutoTable.finalY + 15);

            doc.setFontSize(10);
            doc.setFont("helvetica", "normal");
            doc.text(`Cluster Profile: ${results.cluster_name}`, 10, doc.lastAutoTable.finalY + 22);
            doc.text(`Risk Assessment: ${results.risk_profile}`, 10, doc.lastAutoTable.finalY + 27);

            // Narrative Analysis (The new feature)
            if (results.narrative) {
                doc.setFontSize(11);
                doc.setFont("helvetica", "italic");
                doc.setTextColor(80, 80, 80);
                const splitText = doc.splitTextToSize(results.narrative, 190);
                doc.text(splitText, 10, doc.lastAutoTable.finalY + 40);
            }

            // Action Plan
            const actions = results.actions.map((act, i) => [i + 1, act.replace(/_/g, ' ')]);
            autoTable(doc, {
                startY: results.narrative ? doc.lastAutoTable.finalY + 60 : doc.lastAutoTable.finalY + 45,
                head: [['Step', 'Recommended Intervention']],
                body: actions,
                theme: 'striped',
                headStyles: { fillColor: [20, 184, 166] } // Teal
            });

            doc.save('clinical_report.pdf');
        } catch (err) {
            console.error(err);
            alert("Export Error: " + err.message);
        }
    };

    return (
        <div className="min-h-screen p-4 md:p-8 max-w-[1700px] mx-auto text-neutral-200">

            {/* Top Bar */}
            <div className="flex items-center justify-between mb-8 border-b border-white/5 pb-6">
                <div className="flex items-center gap-4">
                    <button
                        onClick={() => navigate('/')}
                        className="p-2 rounded-lg hover:bg-neutral-800 text-neutral-500 hover:text-white transition-colors"
                    >
                        <ChevronLeft className="w-5 h-5" />
                    </button>
                    <div>
                        <h1 className="text-xl font-bold text-white tracking-tight">Mission Control <span className="text-teal-500">.Med</span></h1>
                        <p className="text-[10px] text-neutral-600 font-mono mt-0.5">SESSION ID: {Math.random().toString(36).substr(2, 9).toUpperCase()}</p>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    {results && (
                        <button
                            onClick={handleExportPDF}
                            className="hidden md:flex items-center gap-2 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-teal-500 font-bold text-xs rounded-lg transition-colors cursor-pointer"
                        >
                            <Download className="w-4 h-4" /> Export Dossier
                        </button>
                    )}
                    <div className="hidden md:flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-900/20 border border-emerald-500/10 text-emerald-500 text-[10px] font-bold tracking-wider">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
                        SYSTEM OPERATIONAL
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

                {/* --- Left Column: Inputs (Visible Panel) --- */}
                <div className="lg:col-span-3 space-y-4">
                    <div className="glass-card p-6 h-[calc(100vh-140px)] overflow-y-auto custom-scrollbar bg-neutral-900/40">
                        <div className="flex items-center gap-2 mb-8 border-b border-white/5 pb-4">
                            <User className="w-4 h-4 text-teal-500" />
                            <h2 className="text-xs font-bold text-neutral-300 uppercase tracking-widest">Patient Vitals</h2>
                        </div>

                        <div className="space-y-8">
                            <div className="space-y-4">
                                <label className="text-[10px] font-bold text-neutral-600 uppercase block mb-3 border-l-2 border-teal-500/50 pl-2">Demographics</label>
                                <div className="grid grid-cols-1 gap-3">
                                    <SelectInput
                                        label="Gender"
                                        value={formData.Gender}
                                        options={[
                                            { label: "Male", value: 1.0 },
                                            { label: "Female", value: 2.0 }
                                        ]}
                                        onChange={(v) => updateField('Gender', v)}
                                    />
                                    <SelectInput
                                        label="Race / Ethnicity"
                                        value={formData.Race}
                                        options={[
                                            { label: "Mexican American", value: 1 },
                                            { label: "Other Hispanic", value: 2 },
                                            { label: "Non-Hispanic White", value: 3 },
                                            { label: "Non-Hispanic Black", value: 4 },
                                            { label: "Multi-Racial", value: 5 }
                                        ]}
                                        onChange={(v) => updateField('Race', v)}
                                    />
                                </div>
                                <RangeInput label="Age" value={formData.Age} min={10} max={90} unit="yrs" onChange={(v) => updateField('Age', v)} />
                            </div>

                            <div className="space-y-4">
                                <label className="text-[10px] font-bold text-neutral-600 uppercase block mb-3 border-l-2 border-teal-500/50 pl-2">Biometrics</label>
                                <RangeInput label="BMI" value={formData.BMI} min={10} max={50} step={0.1} unit="kg/m²" onChange={(v) => updateField('BMI', v)} />
                                <RangeInput label="Glucose" value={formData.Glucose} min={50} max={300} unit="mg/dL" onChange={(v) => updateField('Glucose', v)} />
                                <RangeInput label="Systolic BP" value={formData.SystolicBP} min={80} max={200} unit="mmHg" onChange={(v) => updateField('SystolicBP', v)} />
                            </div>

                            <div className="space-y-4">
                                <label className="text-[10px] font-bold text-neutral-600 uppercase block mb-3 border-l-2 border-teal-500/50 pl-2">Lifestyle</label>
                                <RangeInput label="Calories" value={formData.Calories} min={1000} max={5000} step={50} unit="kcal" onChange={(v) => updateField('Calories', v)} />
                                <RangeInput label="Daily Sugar" value={formData.Sugar} min={0} max={200} unit="g" onChange={(v) => updateField('Sugar', v)} />
                            </div>

                            <button
                                onClick={handlePredict}
                                disabled={loading}
                                className="btn-primary w-full mt-8"
                            >
                                {loading ? (
                                    <Activity className="w-5 h-5 animate-spin" />
                                ) : (
                                    <>
                                        <Zap className="w-4 h-4" /> Run Diagnostics
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>

                {/* --- Main Content: Results & Visuals --- */}
                <div className="lg:col-span-9 space-y-6">

                    {!results ? (
                        <div className="h-full flex flex-col items-center justify-center glass-card p-12 text-center opacity-70 bg-neutral-900/20 border-dashed">
                            <div className="w-20 h-20 bg-neutral-800/50 rounded-full flex items-center justify-center mb-6 animate-pulse">
                                <Activity className="w-8 h-8 text-neutral-600" />
                            </div>
                            <h3 className="text-lg font-bold text-neutral-300 mb-2">Awaiting Input Stream</h3>
                            <p className="text-neutral-500 text-sm max-w-md">Enter patient vitals to initialize clustering engine.</p>
                        </div>
                    ) : (
                        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">

                            {/* Top Stats Row */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                <div className="glass-card p-8 md:col-span-2 flex items-center justify-between relative overflow-hidden bg-neutral-900">
                                    <div className="relative z-10">
                                        <p className="text-neutral-500 text-[10px] font-bold uppercase tracking-widest mb-3">Neural Classification</p>
                                        <h2 className="text-3xl md:text-5xl font-black text-white leading-none tracking-tight">
                                            {results.cluster_name}
                                        </h2>

                                        {/* Narrative Text Display */}
                                        {results.narrative && (
                                            <div className="mt-4 p-4 bg-neutral-800/50 rounded-lg border-l-2 border-teal-500">
                                                <p className="text-xs text-neutral-300 italic leading-relaxed">
                                                    "{results.narrative}"
                                                </p>
                                            </div>
                                        )}

                                        <div className="flex items-center gap-3 mt-6">
                                            <span className="px-2 py-1 bg-neutral-800 rounded text-[10px] text-neutral-400 border border-white/5 font-mono">
                                                ID: {results.cluster}
                                            </span>
                                            {results.risk_profile === 'High' ? (
                                                <span className="flex items-center text-[10px] font-bold text-rose-500 gap-1 bg-rose-950/30 px-2 py-1 rounded border border-rose-900/50">
                                                    <AlertTriangle className="w-3 h-3" /> HIGH RISK DETECTED
                                                </span>
                                            ) : (
                                                <span className="flex items-center text-[10px] font-bold text-emerald-500 gap-1 bg-emerald-950/30 px-2 py-1 rounded border border-emerald-900/50">
                                                    <CheckCircle2 className="w-3 h-3" /> STABLE PROFILE
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                    <div className="absolute right-0 top-0 h-full w-2/3 bg-gradient-to-l from-teal-900/10 to-transparent pointer-events-none"></div>
                                </div>
                                <StatCard
                                    label="Projected BMI (30d)"
                                    value={(formData.BMI + results.predicted_bmi_change).toFixed(1)}
                                    change={results.predicted_bmi_change}
                                    unit="kg/m²"
                                    good={results.predicted_bmi_change < 0}
                                />
                            </div>

                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                                <div className="glass-card p-6 bg-neutral-900/60">
                                    <div className="flex items-center gap-3 mb-6 border-b border-white/5 pb-4">
                                        <div className="w-8 h-8 rounded-lg bg-teal-500/10 flex items-center justify-center text-teal-500">
                                            <Heart className="w-4 h-4" />
                                        </div>
                                        <div>
                                            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Optimization Plan</h3>
                                            <p className="text-[10px] text-neutral-500">AI-Generated Intervention Steps</p>
                                        </div>
                                    </div>

                                    <div className="space-y-3">
                                        {results.actions.map((action, idx) => (
                                            <motion.div
                                                key={idx}
                                                initial={{ opacity: 0, x: -10 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: idx * 0.1 }}
                                                className="p-4 rounded-lg bg-neutral-800/30 border border-white/5 flex items-center gap-4 group hover:bg-neutral-800/50 transition-colors"
                                            >
                                                <div className="w-6 h-6 rounded-full bg-teal-500/10 text-teal-500 flex items-center justify-center text-[10px] font-bold border border-teal-500/20">
                                                    {idx + 1}
                                                </div>
                                                <span className="text-sm font-medium text-neutral-300 capitalize">
                                                    {action.replace(/_/g, ' ')}
                                                </span>
                                                <CheckCircle2 className="w-4 h-4 ml-auto text-emerald-600/50 group-hover:text-emerald-500" />
                                            </motion.div>
                                        ))}
                                        {results.actions.length === 0 && (
                                            <div className="text-center p-12 text-neutral-600 text-sm border-2 border-dashed border-neutral-800 rounded-xl">
                                                No specific interventions required. <br />Patient is within optimal range.
                                            </div>
                                        )}
                                    </div>
                                </div>


                                <div className="glass-card p-6 flex flex-col bg-neutral-900/60">
                                    <div className="flex items-center justify-between mb-6 border-b border-white/5 pb-4">
                                        <div>
                                            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Metabolic Profile</h3>
                                            <p className="text-[10px] text-neutral-500">Multi-Dimensional Analysis</p>
                                        </div>
                                        <Hexagon className="w-4 h-4 text-neutral-600" />
                                    </div>

                                    {/* Charts Container */}
                                    <div className="flex-1 min-h-[250px] grid grid-cols-1 md:grid-cols-2 gap-4">

                                        {/* Radar Chart: Health Footprint */}
                                        <div className="h-full w-full relative">
                                            <p className="text-[10px] text-center text-neutral-600 uppercase font-bold mb-2">Vitals Footprint</p>
                                            <ResponsiveContainer width="100%" height="100%">
                                                <RadarChart outerRadius={65} data={[
                                                    { subject: 'BMI', A: Math.min(100, (formData.BMI / 30) * 100), fullMark: 100 },
                                                    { subject: 'Glucose', A: Math.min(100, (formData.Glucose / 140) * 100), fullMark: 100 },
                                                    { subject: 'BP', A: Math.min(100, (formData.SystolicBP / 140) * 100), fullMark: 100 },
                                                    { subject: 'Cals', A: Math.min(100, (formData.Calories / 3000) * 100), fullMark: 100 },
                                                    { subject: 'Sugar', A: Math.min(100, (formData.Sugar / 100) * 100), fullMark: 100 },
                                                    { subject: 'Activity', A: 40, fullMark: 100 }, // Placeholder
                                                ]}>
                                                    <PolarGrid stroke="rgba(255,255,255,0.05)" />
                                                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#737373', fontSize: 9 }} />
                                                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                                    <Radar name="Patient" dataKey="A" stroke="#14b8a6" strokeWidth={2} fill="#14b8a6" fillOpacity={0.2} />
                                                </RadarChart>
                                            </ResponsiveContainer>
                                        </div>

                                        {/* Area Chart: Trajectory */}
                                        <div className="h-full w-full relative border-l border-white/5 pl-4">
                                            <p className="text-[10px] text-center text-neutral-600 uppercase font-bold mb-2">Projection</p>
                                            <ResponsiveContainer width="100%" height="100%">
                                                <AreaChart data={[
                                                    { step: 'Start', val: 100 },
                                                    { step: 'Step 1', val: 95 },
                                                    { step: 'Step 2', val: 88 },
                                                    { step: 'Step 3', val: 85 },
                                                    { step: 'Target', val: 80 }
                                                ]}>
                                                    <defs>
                                                        <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                                                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.2} />
                                                            <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                                        </linearGradient>
                                                    </defs>
                                                    <XAxis dataKey="step" stroke="#404040" fontSize={8} tickLine={false} axisLine={false} />
                                                    <YAxis hide />
                                                    <Tooltip
                                                        contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid #262626', borderRadius: '4px' }}
                                                        itemStyle={{ color: '#10b981', fontSize: '10px' }}
                                                    />
                                                    <Area type="monotone" dataKey="val" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colorVal)" />
                                                </AreaChart>
                                            </ResponsiveContainer>
                                        </div>

                                    </div>

                                </div>
                            </div>

                        </motion.div>
                    )}
                </div>
            </div>
        </div>
    );
}
