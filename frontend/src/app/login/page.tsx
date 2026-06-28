'use client';

import Navbar from '@/components/Navbar';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';

const loginSchema = z.object({
  email: z.string().email({ message: 'Invalid email address' }),
  password: z.string().min(6, { message: 'Password must be at least 6 characters' }),
});

type LoginFormValues = z.infer<typeof loginSchema>;

export default function LoginPage() {
  const router = useRouter();
  
  const { register, handleSubmit, formState: { errors } } = useForm<LoginFormValues>({
    resolver: zodResolver(loginSchema),
  });

  const onSubmit = (data: LoginFormValues) => {
    console.log('Login credentials:', data);
    alert('Login successful! Redirecting to products page...');
    router.push('/products');
  };

  return (
    <>
      <Navbar />
      <div className="gradient-wash"></div>

      <main className="page-content min-h-screen relative z-10" style={{ paddingTop: '120px' }}>
        <div className="flex justify-center items-center min-h-[calc(100vh-280px)] p-6">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
            className="glass-card w-full max-w-[400px] p-10"
          >
            <div className="mb-6 text-center">
              <h3 className="panel-title text-2xl mb-2">Sign in to Raincast</h3>
              <p className="text-xs text-color-fog-veil">Access your simulator workloads & credentials</p>
            </div>
            
            <form className="flex flex-col gap-4" onSubmit={handleSubmit(onSubmit)}>
              <div className="form-group">
                <label htmlFor="login-email">Email Address</label>
                <input
                  id="login-email"
                  type="email"
                  placeholder="name@company.com"
                  className="form-input"
                  {...register('email')}
                />
                {errors.email && (
                  <span className="text-xs text-color-ember-red mt-1 font-mono">{errors.email.message}</span>
                )}
              </div>
              
              <div className="form-group">
                <label htmlFor="login-password">Password</label>
                <input
                  id="login-password"
                  type="password"
                  placeholder="••••••••"
                  className="form-input"
                  {...register('password')}
                />
                {errors.password && (
                  <span className="text-xs text-color-ember-red mt-1 font-mono">{errors.password.message}</span>
                )}
              </div>
              
              <button type="submit" className="btn-primary w-full mt-2 py-3.5">
                Sign In
              </button>
            </form>
          </motion.div>
        </div>
      </main>

      <footer className="footer-container relative z-10">
        <div className="footer-content">
          <div className="footer-brand">
            <span className="brand-name">Raincast</span>
            <span className="footer-meta font-mono">v2.1.0-API</span>
          </div>
          <div className="footer-copy">
            &copy; 2026 Raincast. Secure credentials authentication.
          </div>
        </div>
      </footer>
    </>
  );
}
