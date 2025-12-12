import React from 'react';

export function RangeInput({ label, name, value, onChange, min, max, step = 1, unit = '' }) {
    return (
        <div className="flex flex-col space-y-2">
            <div className="flex justify-between items-end">
                <label className="text-sm font-medium text-slate-300">{label}</label>
                <div className="flex items-center space-x-2 bg-slate-800/50 rounded px-2 py-1">
                    <input
                        type="number"
                        name={name}
                        value={value}
                        onChange={onChange}
                        className="bg-transparent w-16 text-right outline-none font-mono text-cyan-300 text-sm"
                    />
                    <span className="text-xs font-mono text-slate-500">{unit}</span>
                </div>
            </div>
            <input
                type="range"
                name={name}
                value={value}
                onChange={onChange}
                min={min}
                max={max}
                step={step}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500 hover:accent-blue-400 transition-all"
            />
        </div>
    );
}

export function SelectInput({ label, name, value, onChange, options }) {
    return (
        <div className="flex flex-col space-y-2">
            <label className="text-sm font-medium text-slate-300">{label}</label>
            <select
                name={name}
                value={value}
                onChange={onChange}
                className="glass-input w-full p-2 rounded-lg outline-none"
            >
                {options.map((opt) => (
                    <option key={opt.value} value={opt.value} className="bg-slate-900">
                        {opt.label}
                    </option>
                ))}
            </select>
        </div>
    );
}
