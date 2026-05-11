import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Legend
} from 'recharts';

interface TrendData {
  date: string;
  speed: number;
  stability: number;
}

export default function GaitTrendChart({ data }: { data: TrendData[] }) {
  return (
    <div className="w-full h-[300px] bg-card rounded-xl p-4 border border-border">
      <h3 className="text-textPri font-semibold text-[14px] mb-4">보행 지표 장기 추세 (최근 7일)</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#32425B" />
          <XAxis dataKey="date" stroke="#94A3B8" fontSize={12} />
          <YAxis stroke="#94A3B8" fontSize={12} />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1E293B', border: 'none', borderRadius: '8px' }}
            itemStyle={{ fontSize: '12px' }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="speed" 
            name="보행 속도" 
            stroke="#3B82F6" 
            strokeWidth={2} 
            dot={{ r: 4 }} 
          />
          <Line 
            type="monotone" 
            dataKey="stability" 
            name="안정성 점수" 
            stroke="#10B981" 
            strokeWidth={2} 
            dot={{ r: 4 }} 
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
