import './App.css'
import {BrowserRouter} from "react-router-dom";
import PublicRoutes from "./routes/PublicRoutes.tsx";

function App() {

  return (
      <>
          <BrowserRouter>
            <PublicRoutes />
          </BrowserRouter>


      </>
  )
}

export default App
