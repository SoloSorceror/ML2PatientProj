import React, { useState, useRef } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Upload, FileText, CheckCircle, AlertCircle, ArrowLeft, Download, Database } from 'lucide-react';

export default function BatchUpload() {
    const navigate = useNavigate();
    const fileInputRef = useRef(null);
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setError(null);
            setResults(null);
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setUploading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await axios.post('http://127.0.0.1:8000/batch_predict', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setResults(res.data);
        } catch (err) {
            console.error(err);
            setError("Failed to process batch file. Ensure CSV format is correct.");
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="min-h-screen text-neutral-100 font-sans p-6 md:p-12">
            <motion.div
                initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}
                className="max-w-7xl mx-auto"
            >
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <button
                        onClick={() => navigate('/')}
                        className="flex items-center text-neutral-500 hover:text-white transition-colors text-sm font-semibold mb-2"
                    >
                        <ArrowLeft className="w-4 h-4 mr-2" /> Back to Home
                    </button>
                    <div></div>
                </div>

                <div className="mb-12 text-center">
                    <h1 className="text-3xl md:text-4xl font-extrabold text-white">Batch Analysis <span className="text-neutral-600">Pro</span></h1>
                    <div className="h-1 w-24 bg-teal-500 mx-auto mt-4 rounded-full"></div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                    {/* Upload Section */}
                    <div className="lg:col-span-4 space-y-6">
                        <div className="glass-card p-8 text-center space-y-6">
                            <div className="w-16 h-16 bg-neutral-800/50 rounded-2xl flex items-center justify-center mx-auto border border-white/5">
                                <Upload className="w-8 h-8 text-teal-500" />
                            </div>
                            <div>
                                <h2 className="text-xl font-bold text-white mb-2">Upload Dataset</h2>
                                <p className="text-sm text-neutral-400 mb-6">Upload CSV datasets for bulk segmentation.</p>
                            </div>

                            <input
                                type="file"
                                accept=".csv"
                                ref={fileInputRef}
                                onChange={handleFileChange}
                                className="hidden"
                            />

                            <button
                                onClick={() => fileInputRef.current?.click()}
                                className={`w-full border-2 border-dashed rounded-xl p-8 mb-4 transition-all cursor-pointer group ${file ? 'border-teal-500/50 bg-teal-500/5' : 'border-neutral-800 hover:border-neutral-600 hover:bg-neutral-900'}`}
                            >
                                <FileText className={`w-8 h-8 mx-auto mb-2 transition-colors ${file ? 'text-teal-400' : 'text-neutral-600 group-hover:text-neutral-400'}`} />
                                <span className="text-xs font-mono text-neutral-400 truncate block">
                                    {file ? file.name : "Select CSV File"}
                                </span>
                            </button>

                            <button
                                onClick={handleUpload}
                                disabled={!file || uploading}
                                className={`w-full py-4 rounded-xl font-bold transition-all flex items-center justify-center ${!file || uploading ? 'bg-neutral-800 text-neutral-600 cursor-not-allowed' : 'btn-primary'}`}
                            >
                                {uploading ? "Processing..." : "Run Global Analysis"}
                            </button>

                            {error && (
                                <div className="p-3 bg-red-900/10 border border-red-500/20 rounded-lg text-red-400 text-xs flex items-center justify-center">
                                    <AlertCircle className="w-3 h-3 mr-2" /> {error}
                                </div>
                            )}
                        </div>

                        {/* Requirements Card */}
                        <div className="p-6 rounded-xl border border-dashed border-neutral-800">
                            <h3 className="text-xs font-bold text-neutral-500 uppercase mb-3">Schema Requirements</h3>
                            <ul className="text-xs text-neutral-600 space-y-2 font-mono">
                                <li>• Age, Gender (1=M, 2=F)</li>
                                <li>• BMI, SBP, DBP, Glucose</li>
                                <li>• Calories, Sugar, Fat</li>
                            </ul>
                        </div>
                    </div>

                    {/* Results Section */}
                    <div className="lg:col-span-8">
                        <div className="glass-card min-h-[600px] relative flex flex-col overflow-hidden">
                            {!results ? (
                                <div className="absolute inset-0 flex flex-col items-center justify-center text-neutral-700">
                                    <FileText className="w-16 h-16 mb-4 opacity-20" />
                                    <p className="font-mono text-sm">Waiting for data stream...</p>
                                </div>
                            ) : (
                                <>
                                    <div className="p-6 border-b border-white/5 flex justify-between items-center bg-neutral-950/30">
                                        <h2 className="text-lg font-bold text-white flex items-center gap-2">
                                            <CheckCircle className="w-5 h-5 text-teal-500" /> Analysis Complete
                                        </h2>
                                        <div className="flex items-center gap-4">
                                            <span className="px-3 py-1 bg-neutral-800 rounded text-xs text-neutral-400 border border-white/5">{results.length} Records</span>
                                            <button className="text-xs font-bold text-teal-500 hover:text-teal-400 flex items-center transition-colors">
                                                <Download className="w-4 h-4 mr-1" /> Export Report
                                            </button>
                                        </div>
                                    </div>

                                    <div className="overflow-auto flex-1 custom-scrollbar p-0">
                                        <table className="w-full text-left border-collapse">
                                            <thead className="bg-neutral-950/50 text-xs font-bold text-neutral-500 uppercase tracking-wider sticky top-0 backdrop-blur-md z-10">
                                                <tr>
                                                    <th className="p-5 border-b border-white/5">ID</th>
                                                    <th className="p-5 border-b border-white/5">Cluster ID</th>
                                                    <th className="p-5 border-b border-white/5">Neural Classification</th>
                                                    <th className="p-5 border-b border-white/5 text-right">Risk Profile</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-white/5">
                                                {results.map((row, idx) => (
                                                    <motion.tr
                                                        initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: idx * 0.05 }}
                                                        key={idx}
                                                        className="hover:bg-white/[0.02] transition-colors group"
                                                    >
                                                        <td className="p-5 text-neutral-500 font-mono text-sm">#{row.seqn}</td>
                                                        <td className="p-5 text-neutral-400 font-mono">{row.cluster}</td>
                                                        <td className="p-5">
                                                            <span className="font-medium text-neutral-200">{row.cluster_name}</span>
                                                        </td>
                                                        <td className="p-5 text-right">
                                                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${row.risk_profile === 'High'
                                                                    ? 'bg-red-500/10 text-red-500 border border-red-500/20'
                                                                    : 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20'
                                                                }`}>
                                                                {row.risk_profile}
                                                            </span>
                                                        </td>
                                                    </motion.tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                </div>
            </motion.div>
        </div>
    );
}
