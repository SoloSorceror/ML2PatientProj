import React from 'react';

export function InputGroup({ title, children }) {
    return (
        <div className="glass-panel p-6 rounded-xl mb-6">
            <h3 className="text-xl font-bold mb-4 text-blue-400 border-b border-white/10 pb-2">{title}</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {children}
            </div>
        </div>
    );
}
