import { useState, useMemo } from 'react';
import {
  Brain,
  Search,
  Hash,
  Layers,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  Sparkles,
  GitFork,
  Filter,
} from 'lucide-react';
import { OntologyData, OntologyConcept } from '../types';
import { Button } from './ui/Button';

interface OntologyTabProps {
  ontology: OntologyData | null;
  onRefresh: () => void;
  isLoading: boolean;
}

export function OntologyTab({ ontology, onRefresh, isLoading }: OntologyTabProps) {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedTier, setSelectedTier] = useState<string>('ALL');
  const [selectedConcept, setSelectedConcept] = useState<OntologyConcept | null>(null);

  const concepts = ontology?.concepts || [];
  const relations = ontology?.relations || [];
  const summary = ontology?.summary || {
    total_concepts: 0,
    total_relations: 0,
    asserted_count: 0,
    induced_enforcing_count: 0,
    induced_candidate_count: 0,
  };

  const filteredConcepts = useMemo(() => {
    return concepts.filter((c) => {
      const matchesTier =
        selectedTier === 'ALL' || c.tier.toLowerCase() === selectedTier.toLowerCase();
      const term = searchTerm.toLowerCase();
      const matchesSearch =
        !term ||
        c.name.toLowerCase().includes(term) ||
        (c.parent_type && c.parent_type.toLowerCase().includes(term)) ||
        Object.keys(c.attributes).some((k) => k.toLowerCase().includes(term));
      return matchesTier && matchesSearch;
    });
  }, [concepts, selectedTier, searchTerm]);

  const activeConcept = selectedConcept || concepts[0] || null;

  return (
    <div className="space-y-6">
      {/* Overview & Tier Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Total Concepts</span>
            <Brain className="w-4 h-4 text-cyan-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-white">{summary.total_concepts}</p>
          <p className="text-[11px] text-cyan-400 mt-0.5 font-mono">LinkML schemas</p>
        </div>

        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Asserted (Axiomatic)</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-emerald-400">{summary.asserted_count}</p>
          <p className="text-[11px] text-slate-400 mt-0.5 font-mono">Precedence: 100 (Immutable)</p>
        </div>

        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Induced Enforcing</span>
            <Sparkles className="w-4 h-4 text-blue-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-blue-400">
            {summary.induced_enforcing_count}
          </p>
          <p className="text-[11px] text-slate-400 mt-0.5 font-mono">Precedence: 50 (Runtime)</p>
        </div>

        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Induced Candidate</span>
            <GitFork className="w-4 h-4 text-amber-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-amber-400">
            {summary.induced_candidate_count}
          </p>
          <p className="text-[11px] text-slate-400 mt-0.5 font-mono">Precedence: 10 (Advisory)</p>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg flex flex-wrap gap-3">
        <div className="relative grow basis-48">
          <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
          <input
            type="text"
            placeholder="Search concepts by name, parent type, or attribute schema..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 shadow-inner"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Filter className="w-3.5 h-3.5 text-slate-500" />
          <select
            value={selectedTier}
            onChange={(e) => setSelectedTier(e.target.value)}
            className="bg-slate-950 border border-slate-800 text-xs text-slate-300 rounded-xl px-3 py-2 focus:outline-none focus:border-cyan-500 font-mono shadow-inner"
          >
            <option value="ALL">All Tiers</option>
            <option value="asserted">Tier: Asserted (100)</option>
            <option value="induced_enforcing">Tier: Induced Enforcing (50)</option>
            <option value="induced_candidate">Tier: Induced Candidate (10)</option>
          </select>

          <Button
            variant="bordered"
            onClick={onRefresh}
            disabled={isLoading}
            className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border-slate-700 text-slate-200 hover:text-slate-200 whitespace-nowrap"
          >
            {isLoading ? 'Syncing...' : 'Sync Graph'}
          </Button>
        </div>
      </div>

      {/* Concepts Grid & Detail Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Concepts List */}
        <div className="p-5 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg space-y-3">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              LinkML Concept Registry
            </h3>
            <span className="text-[10px] font-mono text-slate-400">
              {filteredConcepts.length} concepts
            </span>
          </div>

          <div className="space-y-2.5 max-h-[500px] overflow-y-auto pr-1">
            {filteredConcepts.map((concept) => {
              const isSelected = activeConcept?.name === concept.name;
              const isAsserted = concept.tier === 'asserted';
              const isEnforcing = concept.tier === 'induced_enforcing';

              return (
                <div
                  key={concept.name}
                  onClick={() => setSelectedConcept(concept)}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-500/80 shadow-md shadow-cyan-950/30'
                      : 'bg-slate-950/70 border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-white">
                      {concept.name}
                    </span>
                    <span
                      className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded ${
                        isAsserted
                          ? 'bg-emerald-950 text-emerald-300 border border-emerald-800/60'
                          : isEnforcing
                          ? 'bg-blue-950 text-blue-300 border border-blue-800/60'
                          : 'bg-amber-950 text-amber-300 border border-amber-800/60'
                      }`}
                    >
                      {concept.tier} [{concept.precedence}]
                    </span>
                  </div>

                  {concept.parent_type && (
                    <p className="text-[11px] text-slate-400 mt-1 flex items-center gap-1">
                      <span>Inherits from:</span>
                      <strong className="text-slate-300 font-mono">{concept.parent_type}</strong>
                    </p>
                  )}

                  <div className="mt-2 flex items-center justify-between text-[10px] text-slate-500 font-mono">
                    <span>{Object.keys(concept.attributes).length} attributes</span>
                    <span>SHA-256: {concept.content_hash.slice(0, 10)}...</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Selected Concept Schema & Provenance Detail */}
        <div className="lg:col-span-2 p-5 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <Brain className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-semibold text-white">
                Concept Schema & Provenance Integrity
              </h3>
            </div>
            {activeConcept && (
              <span className="text-[10px] font-mono bg-slate-950 text-slate-300 border border-slate-800 px-2 py-0.5 rounded">
                Provenance Origin: {activeConcept.provenance.origin}
              </span>
            )}
          </div>

          {activeConcept ? (
            <div className="space-y-4 text-xs">
              <div className="p-4 bg-slate-950/80 border border-slate-800 rounded-xl space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-base font-bold text-cyan-300">
                    {activeConcept.name}
                  </span>
                  <span className="flex items-center gap-1 text-emerald-400 text-[11px]">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Hash Verified
                  </span>
                </div>

                <div className="flex items-center gap-2 text-[11px] font-mono text-slate-400">
                  <Hash className="w-3.5 h-3.5 text-slate-500" />
                  <span>Canonical SHA-256:</span>
                  <code className="text-cyan-400 bg-slate-900 px-1.5 py-0.5 rounded">
                    {activeConcept.content_hash}
                  </code>
                </div>
              </div>

              {/* Attributes Schema */}
              <div>
                <h4 className="text-xs font-semibold text-slate-200 mb-2">Attributes & Types</h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {Object.entries(activeConcept.attributes).map(([attr, type]) => {
                    const isRequired = activeConcept.required_fields.includes(attr);
                    return (
                      <div
                        key={attr}
                        className="p-2.5 bg-slate-950/60 border border-slate-800/80 rounded-lg flex items-center justify-between"
                      >
                        <div className="flex items-center gap-1.5 font-mono">
                          <span className="text-slate-200">{attr}</span>
                          {isRequired && (
                            <span className="text-[9px] bg-rose-950 text-rose-400 border border-rose-800/50 px-1 rounded">
                              req
                            </span>
                          )}
                        </div>
                        <span className="text-[10px] text-cyan-400 font-mono">{type}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Active Relations Involving Concept */}
              <div>
                <h4 className="text-xs font-semibold text-slate-200 mb-2">Active Relations (LinkML)</h4>
                <div className="space-y-2">
                  {relations
                    .filter(
                      (r) =>
                        r.source === activeConcept.name || r.target === activeConcept.name
                    )
                    .map((rel) => (
                      <div
                        key={rel.id}
                        className="p-3 bg-slate-950/60 border border-slate-800/80 rounded-xl flex flex-wrap items-center justify-between gap-2"
                      >
                        <div className="flex items-center gap-2 font-mono text-xs">
                          <span className="text-cyan-300 font-semibold">{rel.source}</span>
                          <ArrowRight className="w-3.5 h-3.5 text-slate-500" />
                          <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-purple-300">
                            {rel.predicate}
                          </span>
                          <ArrowRight className="w-3.5 h-3.5 text-slate-500" />
                          <span className="text-cyan-300 font-semibold">{rel.target}</span>
                        </div>

                        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
                          <span className="bg-slate-900 px-1.5 py-0.5 rounded">
                            Conf: {Math.round(rel.confidence * 100)}%
                          </span>
                          <span className="text-slate-500">
                            Hash: {rel.content_hash.slice(0, 8)}...
                          </span>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="p-10 text-center text-slate-500 text-xs">
              Select a concept from the list to view its schema and relations.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
