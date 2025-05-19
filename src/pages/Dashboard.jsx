// src/pages/Dashboard.jsx
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { BarChart3, FileText, MessagesSquare, Clock } from 'lucide-react';
import Card from '../components/common/Card';
import useStore from '../store/useStore';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const Dashboard = () => {
  const { t } = useTranslation();
  const { stats, setStats } = useStore();
  
  useEffect(() => {
    // In a real app, you would fetch this data from your backend
    setStats({
      documentCount: 12,
      queryCount: 45,
      avgResponseTime: 0.8,
      activityData: [
        { name: 'Mon', queries: 4 },
        { name: 'Tue', queries: 6 },
        { name: 'Wed', queries: 8 },
        { name: 'Thu', queries: 5 },
        { name: 'Fri', queries: 9 },
        { name: 'Sat', queries: 3 },
        { name: 'Sun', queries: 2 },
      ]
    });
  }, [setStats]);
  
  // Ensure stats has default values if not yet loaded
  const safeStats = {
    documentCount: stats?.documentCount || 0,
    queryCount: stats?.queryCount || 0,
    avgResponseTime: stats?.avgResponseTime || 0,
    activityData: stats?.activityData || []
  };
  
  const statsCards = [
    {
      title: t('dashboard.totalDocuments'),
      value: safeStats.documentCount,
      icon: <FileText size={24} className="text-primary" />,
      change: '+3 this week',
      positive: true
    },
    {
      title: t('dashboard.totalQueries'),
      value: safeStats.queryCount,
      icon: <MessagesSquare size={24} className="text-secondary" />,
      change: '+12 this week',
      positive: true
    },
    {
      title: t('dashboard.avgResponseTime'),
      value: `${safeStats.avgResponseTime}s`,
      icon: <Clock size={24} className="text-accent" />,
      change: '-0.2s from last week',
      positive: true
    },
  ];
  
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">{t('dashboard.title')}</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Overview of your documents and interactions
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {statsCards.map((card, index) => (
          <Card key={index} className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">{card.title}</p>
                <h3 className="text-2xl font-bold mt-1">{card.value}</h3>
                <p className={`text-xs mt-1 ${card.positive ? 'text-green-500' : 'text-red-500'}`}>
                  {card.change}
                </p>
              </div>
              <div className="p-3 bg-gray-100 dark:bg-gray-700 rounded-full">
                {card.icon}
              </div>
            </div>
          </Card>
        ))}
      </div>
      
      <Card className="p-6">
        <h3 className="text-lg font-medium mb-4">Activity</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={safeStats.activityData}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="queries" fill="#4f46e5" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
};

export default Dashboard;