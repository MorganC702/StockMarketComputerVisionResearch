// src/routes/AppRoutes.tsx
import { Routes, Route } from 'react-router-dom'
import CandlestickChart from '../charts/views/CandlestickChartView'
import PublicLayout from "../layout/PublicLayout";

export default function AppRoutes() {
    return (
        <Routes>
            <Route element={<PublicLayout />}>

            <Route path="/" element={<CandlestickChart/>} />
            </Route>
        </Routes>
    )
}
