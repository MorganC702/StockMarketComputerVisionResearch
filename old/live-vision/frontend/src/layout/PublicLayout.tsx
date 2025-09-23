import React from 'react';
import AppNavigation from './PublicNavigation';
import { Outlet } from 'react-router-dom';
import Footer from './Footer';

const PublicLayout: React.FC = () => {
    return (
        <AppNavigation
            inner_content={
                <>
                    <main>
                        <Outlet />
                    </main>
                    <Footer />
                </>
            }
        />
    );
};

export default PublicLayout;
