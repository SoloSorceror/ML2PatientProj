import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, FileText } from 'lucide-react';

export default function Home() {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen relative overflow-hidden flex flex-col items-center justify-center p-6 sm:p-12">

            {/* Subtle Background Mesh */}
            <div className="absolute inset-0 z-0 pointer-events-none">
                <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-neutral-800/20 via-transparent to-transparent opacity-50"></div>
            </div>

            <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="relative z-10 max-w-5xl w-full text-center"
            >
                {/* Minimal Badge */}
                <motion.div
                    whileHover={{ scale: 1.02 }}
                    className="inline-flex items-center gap-3 px-4 py-2 rounded-full border border-neutral-800 bg-neutral-900/50 text-teal-500 text-xs font-bold tracking-widest uppercase mb-6 backdrop-blur-sm"
                >
                    <span className="w-2 h-2 rounded-full bg-teal-500 animate-pulse"></span>
                    Clinical Grade Analytics
                </motion.div>

                <h1 className="text-6xl md:text-8xl font-black mb-8 tracking-tight text-white leading-tight">
                    CORTEX <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-emerald-500">MED</span>
                </h1>

                <p className="text-xl md:text-2xl text-neutral-400 font-light max-w-2xl mx-auto leading-relaxed mb-12">
                    Next-generation patient segmentation & <span className="text-neutral-200 font-medium">metabolic optimization engine</span>.
                    Precision medicine powered by unsupervised learning.
                </p>

                {/* Main Actions */}
                <div className="flex flex-col sm:flex-row gap-6 justify-center items-center mb-20">
                    <button
                        onClick={() => navigate('/dashboard')}
                        className="btn-primary min-w-[200px] h-[56px] text-lg hover:shadow-teal-500/20"
                    >
                        Initialize System <ArrowRight className="w-5 h-5" />
                    </button>

                    <button
                        onClick={() => navigate('/batch')}
                        className="btn-secondary min-w-[200px] h-[56px] text-lg"
                    >
                        <FileText className="w-5 h-5 text-neutral-500" /> Batch Analysis
                    </button>
                </div>

                {/* Feature Grid (Clean) */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
                    {[
                        { title: "Neural Clustering", text: "Unsupervised UMAP + KMeans segmentation of 9 distinct metabolic profiles." },
                        { title: "Trajectory Sim", text: "Predictive modeling of biometric outcomes based on lifestyle interventions." },
                        { title: "Risk Stratification", text: "Real-time identification of at-risk groups using multi-dimensional analysis." }
                    ].map((item, i) => (
                        <motion.div
                            key={i}
                            whileHover={{ y: -5 }}
                            className="p-6 rounded-xl border border-white/5 bg-neutral-900/30 hover:bg-neutral-900/50 transition-colors"
                        >
                            <h3 className="text-lg font-bold text-white mb-2">{item.title}</h3>
                            <p className="text-sm text-neutral-500 leading-relaxed">{item.text}</p>
                        </motion.div>
                    ))}
                </div>
            </motion.div>

            {/* Footer */}
            <div className="absolute bottom-6 flex gap-4 text-xs font-mono text-neutral-700">
                <span>V 3.5.0 (SAFE-MODE)</span>
                <span>•</span>
                <span>SYSTEM READY</span>
            </div>
        </div>
    );
}
