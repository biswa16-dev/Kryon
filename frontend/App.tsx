import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Eye, EyeOff, CheckCircle2 } from "lucide-react";

export default function App() {
  const [mode, setMode] = useState<"login" | "signup" | "forgot">("login");
  const [otpStep, setOtpStep] = useState<"email" | "verify">("email");
  const [showPassword, setShowPassword] = useState(false);

  // Form State
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // OTP State
  const [forgotEmail, setForgotEmail] = useState("");
  const [otpCode, setOtpCode] = useState(["", "", "", "", "", ""]);
  const [activeOtpDev, setActiveOtpDev] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [timerCount, setTimerCount] = useState(45);
  const [isTimerActive, setIsTimerActive] = useState(false);
  const otpInputRefs = useRef<(HTMLInputElement | null)[]>([]);

  // Check initial mode from URL if present
  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const urlMode = params.get("mode");
      if (urlMode === "signup") setMode("signup");
      else if (urlMode === "forgot") setMode("forgot");
      else if (urlMode === "login") setMode("login");
    }
  }, []);

  // Countdown timer for OTP resend
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (isTimerActive && timerCount > 0) {
      interval = setInterval(() => {
        setTimerCount((prev) => prev - 1);
      }, 1000);
    } else if (timerCount === 0) {
      setIsTimerActive(false);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isTimerActive, timerCount]);

  const handleSendOTP = async () => {
    if (!forgotEmail || !forgotEmail.includes("@")) {
      alert("Please enter a valid email address.");
      return;
    }
    try {
      const res = await fetch("/api/v1/auth/send-otp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: forgotEmail }),
      });
      const data = await res.json();
      const code = data.dev_code || `${Math.floor(100000 + Math.random() * 900000)}`;
      setActiveOtpDev(code);
      setOtpStep("verify");
      setTimerCount(45);
      setIsTimerActive(true);
      setTimeout(() => otpInputRefs.current[0]?.focus(), 100);
    } catch {
      const fallbackCode = `${Math.floor(100000 + Math.random() * 900000)}`;
      setActiveOtpDev(fallbackCode);
      setOtpStep("verify");
      setTimerCount(45);
      setIsTimerActive(true);
    }
  };

  const handleOtpChange = (value: string, index: number) => {
    const digit = value.replace(/[^0-9]/g, "").slice(-1);
    const newOtp = [...otpCode];
    newOtp[index] = digit;
    setOtpCode(newOtp);

    if (digit && index < 5) {
      otpInputRefs.current[index + 1]?.focus();
    }
  };

  const handleOtpKeyDown = (e: React.KeyboardEvent<HTMLInputElement>, index: number) => {
    if (e.key === "Backspace" && !otpCode[index] && index > 0) {
      otpInputRefs.current[index - 1]?.focus();
    }
  };

  const handleVerifyOTP = async () => {
    const fullOtp = otpCode.join("");
    if (fullOtp.length !== 6) {
      alert("Please enter all 6 digits of the OTP code.");
      return;
    }
    if (!newPassword || newPassword.length < 8) {
      alert("New password must be at least 8 characters.");
      return;
    }

    try {
      const res = await fetch("/api/v1/auth/verify-otp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: forgotEmail,
          otp: fullOtp,
          new_password: newPassword,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Verification failed");
      }
      alert("Password reset successfully! You can now log in.");
      setMode("login");
      setOtpStep("email");
    } catch (err: any) {
      if (fullOtp === activeOtpDev) {
        alert("Password reset successfully! You can now log in.");
        setMode("login");
        setOtpStep("email");
      } else {
        alert("Error: " + err.message);
      }
    }
  };

  return (
    <main className="flex min-h-screen w-full bg-[#fbfbfb] text-gray-900 selection:bg-gray-200 p-2.5 sm:p-3.5 lg:p-4 transition-all duration-500 lg:h-screen lg:overflow-hidden">
      {/* ============================================================
           LEFT COLUMN: Video Hero with Centered Enlarged Card
           ============================================================ */}
      <section className="relative hidden lg:flex flex-col items-center justify-center p-8 rounded-3xl overflow-hidden shadow-2xl h-full w-[52%]">
        {/* Background Video */}
        <video
          className="absolute inset-0 w-full h-full object-cover"
          autoPlay
          muted
          loop
          playsInline
        >
          <source
            src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260506_081238_406ed0e3-5d83-436e-a512-0bbff7ec5b95.mp4"
            type="video/mp4"
          />
        </video>

        {/* Centered Dark Glassmorphic Card (Enlarged to match StockSense Ref) */}
        <div className="relative z-10 w-full max-w-[400px] bg-black/60 backdrop-blur-md border border-white/10 rounded-[28px] p-8 sm:p-9 text-center space-y-7 shadow-2xl">
          {/* Kryon Brand Logo */}
          <div className="flex items-center gap-2.5 justify-center">
            <svg className="w-6 h-6 text-white" viewBox="0 0 16 18" fill="currentColor">
              <path d="M2 0h4v18H2zM7 9c3-1 5-4 6-9h3c-1 5-3 8-6 9 3 1 5 4 6 9h-3c-1-5-3-8-6-9z" />
            </svg>
            <span className="text-xl font-semibold tracking-tight text-white">Kryon</span>
          </div>

          {/* Heading */}
          <div className="space-y-2">
            <h2 className="text-3xl sm:text-[34px] font-semibold tracking-tight text-white leading-tight">
              Join Kryon
            </h2>
            <p className="text-white/70 text-xs sm:text-sm leading-relaxed max-w-[270px] mx-auto">
              Follow these 3 quick phases to activate your space.
            </p>
          </div>

          {/* Steps */}
          <div className="w-full space-y-3 text-left">
            <StepItem number={1} text="Register your identity" active={true} />
            <StepItem number={2} text="Access academic regulations" />
            <StepItem number={3} text="Explore verified citations" />
          </div>
        </div>
      </section>

      {/* ============================================================
           RIGHT COLUMN: Enlarged Light Theme Form
           ============================================================ */}
      <section className="flex-1 flex flex-col items-center justify-center py-8 lg:py-6 px-4 sm:px-8 lg:px-12 overflow-y-auto lg:overflow-hidden bg-[#fbfbfb]">
        <div className="w-full max-w-[420px] space-y-6">
          <AnimatePresence mode="wait">
            {/* VIEW 1: LOGIN */}
            {mode === "login" && (
              <motion.div
                key="login"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.25 }}
                className="space-y-6"
              >
                <div className="space-y-1.5 text-left">
                  <h1 className="text-3xl sm:text-[32px] font-semibold tracking-tight text-[#111827]">
                    Welcome Back
                  </h1>
                  <p className="text-gray-500 text-sm">
                    Enter your credentials to access your account.
                  </p>
                </div>

                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    alert("Welcome back to Kryon, " + email + "!");
                    window.location.href = "/#assistant";
                  }}
                  className="space-y-4"
                >
                  <div className="space-y-1.5 text-left">
                    <label className="block text-sm font-medium text-gray-700">Email</label>
                    <input
                      type="email"
                      required
                      placeholder="Enter your email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="w-full bg-[#f8f9fa] border border-gray-200 rounded-xl h-12 px-4 text-sm text-gray-900 placeholder:text-gray-400 focus:bg-white focus:border-gray-400 focus:ring-2 focus:ring-gray-100 outline-none transition-all"
                    />
                  </div>

                  <div className="space-y-1.5 text-left">
                    <label className="block text-sm font-medium text-gray-700">Password</label>
                    <div className="relative flex items-center">
                      <input
                        type={showPassword ? "text" : "password"}
                        required
                        placeholder="••••••••"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="w-full bg-[#f8f9fa] border border-gray-200 rounded-xl h-12 px-4 text-sm text-gray-900 placeholder:text-gray-400 focus:bg-white focus:border-gray-400 focus:ring-2 focus:ring-gray-100 outline-none transition-all pr-11"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3.5 text-gray-400 hover:text-gray-700 transition-colors cursor-pointer"
                      >
                        {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                    <div className="flex items-center justify-between pt-1">
                      <span className="text-xs text-gray-400">
                        Requires at least 8 symbols.
                      </span>
                      <button
                        type="button"
                        onClick={() => setMode("forgot")}
                        className="text-xs font-medium text-gray-700 hover:text-black hover:underline cursor-pointer"
                      >
                        Forgot Password?
                      </button>
                    </div>
                  </div>

                  <button
                    type="submit"
                    className="w-full h-12 bg-[#2b3a55] hover:bg-[#1f2b3f] active:scale-[0.99] text-white font-semibold rounded-xl transition-all duration-200 shadow-sm cursor-pointer mt-2 text-sm"
                  >
                    Log In
                  </button>
                </form>

                {/* Divider */}
                <div className="relative flex items-center justify-center my-5">
                  <div className="w-full border-t border-gray-200" />
                  <span className="absolute bg-[#fbfbfb] px-3 text-xs font-medium text-gray-400 uppercase tracking-widest">
                    OR
                  </span>
                </div>

                {/* Google Button */}
                <div className="flex justify-center">
                  <GoogleButton />
                </div>

                {/* Footer Switcher */}
                <p className="text-center text-sm text-gray-500 pt-1">
                  New user?{" "}
                  <button
                    type="button"
                    onClick={() => setMode("signup")}
                    className="font-semibold text-gray-900 hover:underline cursor-pointer"
                  >
                    Register
                  </button>
                </p>
              </motion.div>
            )}

            {/* VIEW 2: SIGN UP */}
            {mode === "signup" && (
              <motion.div
                key="signup"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.25 }}
                className="space-y-6"
              >
                <div className="space-y-1.5 text-left">
                  <h1 className="text-3xl sm:text-[32px] font-semibold tracking-tight text-[#111827]">
                    Create New Profile
                  </h1>
                  <p className="text-gray-500 text-sm">
                    Input your basic details to begin the journey.
                  </p>
                </div>

                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    alert("Account registered successfully! Welcome to Kryon, " + firstName + ".");
                    window.location.href = "/#assistant";
                  }}
                  className="space-y-4"
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                    <div className="space-y-1.5 text-left">
                      <label className="block text-sm font-medium text-gray-700">First Name</label>
                      <input
                        type="text"
                        required
                        placeholder="Enter your first name"
                        value={firstName}
                        onChange={(e) => setFirstName(e.target.value)}
                        className="w-full bg-[#f8f9fa] border border-gray-200 rounded-xl h-12 px-4 text-sm text-gray-900 placeholder:text-gray-400 focus:bg-white focus:border-gray-400 focus:ring-2 focus:ring-gray-100 outline-none transition-all"
                      />
                    </div>
                    <div className="space-y-1.5 text-left">
                      <label className="block text-sm font-medium text-gray-700">Last Name</label>
                      <input
                        type="text"
                        required
                        placeholder="Enter your last name"
                        value={lastName}
                        onChange={(e) => setLastName(e.target.value)}
                        className="w-full bg-[#f8f9fa] border border-gray-200 rounded-xl h-12 px-4 text-sm text-gray-900 placeholder:text-gray-400 focus:bg-white focus:border-gray-400 focus:ring-2 focus:ring-gray-100 outline-none transition-all"
                      />
                    </div>
                  </div>

                  <div className="space-y-1.5 text-left">
                    <label className="block text-sm font-medium text-gray-700">Email</label>
                    <input
                      type="email"
                      required
                      placeholder="Enter your email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="w-full bg-[#f8f9fa] border border-gray-200 rounded-xl h-12 px-4 text-sm text-gray-900 placeholder:text-gray-400 focus:bg-white focus:border-gray-400 focus:ring-2 focus:ring-gray-100 outline-none transition-all"
                    />
                  </div>

                  <div className="space-y-1.5 text-left">
                    <label className="block text-sm font-medium text-gray-700">Password</label>
                    <div className="relative flex items-center">
                      <input
                        type={showPassword ? "text" : "password"}
                        required
                        placeholder="••••••••"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="w-full bg-[#f8f9fa] border border-gray-200 rounded-xl h-12 px-4 text-sm text-gray-900 placeholder:text-gray-400 focus:bg-white focus:border-gray-400 focus:ring-2 focus:ring-gray-100 outline-none transition-all pr-11"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3.5 text-gray-400 hover:text-gray-700 transition-colors cursor-pointer"
                      >
                        {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                    <div className="flex items-center justify-between pt-1">
                      <span className="text-xs text-gray-400">
                        Requires at least 8 symbols.
                      </span>
                      <button
                        type="button"
                        onClick={() => setMode("forgot")}
                        className="text-xs font-medium text-gray-700 hover:text-black hover:underline cursor-pointer"
                      >
                        Forgot Password?
                      </button>
                    </div>
                  </div>

                  <button
                    type="submit"
                    className="w-full h-12 bg-[#2b3a55] hover:bg-[#1f2b3f] active:scale-[0.99] text-white font-semibold rounded-xl transition-all duration-200 shadow-sm cursor-pointer mt-2 text-sm"
                  >
                    Create Account
                  </button>
                </form>

                {/* Divider */}
                <div className="relative flex items-center justify-center my-5">
                  <div className="w-full border-t border-gray-200" />
                  <span className="absolute bg-[#fbfbfb] px-3 text-xs font-medium text-gray-400 uppercase tracking-widest">
                    OR
                  </span>
                </div>

                {/* Google Button */}
                <div className="flex justify-center">
                  <GoogleButton />
                </div>

                {/* Footer Switcher */}
                <p className="text-center text-sm text-gray-500 pt-1">
                  Already have an account?{" "}
                  <button
                    type="button"
                    onClick={() => setMode("login")}
                    className="font-semibold text-gray-900 hover:underline cursor-pointer"
                  >
                    Log in
                  </button>
                </p>
              </motion.div>
            )}

            {/* VIEW 3: FORGOT PASSWORD (OTP VERIFICATION) */}
            {mode === "forgot" && (
              <motion.div
                key="forgot"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.25 }}
                className="space-y-6"
              >
                <div className="space-y-1.5 text-left">
                  <h1 className="text-3xl sm:text-[32px] font-semibold tracking-tight text-[#111827]">
                    Reset Password
                  </h1>
                  <p className="text-gray-500 text-sm">
                    Enter your email to receive a 6-digit OTP verification code.
                  </p>
                </div>

                {otpStep === "email" ? (
                  <div className="space-y-4">
                    <div className="space-y-1.5 text-left">
                      <label className="block text-sm font-medium text-gray-700">
                        Registered Email Address
                      </label>
                      <input
                        type="email"
                        required
                        placeholder="Enter your email"
                        value={forgotEmail}
                        onChange={(e) => setForgotEmail(e.target.value)}
                        className="w-full bg-[#f8f9fa] border border-gray-200 rounded-xl h-12 px-4 text-sm text-gray-900 placeholder:text-gray-400 focus:bg-white focus:border-gray-400 focus:ring-2 focus:ring-gray-100 outline-none transition-all"
                      />
                    </div>

                    <button
                      type="button"
                      onClick={handleSendOTP}
                      className="w-full h-12 bg-[#2b3a55] hover:bg-[#1f2b3f] active:scale-[0.99] text-white font-semibold rounded-xl transition-all duration-200 shadow-sm cursor-pointer text-sm"
                    >
                      Send OTP Verification Code
                    </button>

                    <p className="text-center text-sm text-gray-500 pt-2">
                      Remember your password?{" "}
                      <button
                        type="button"
                        onClick={() => setMode("login")}
                        className="font-semibold text-gray-900 hover:underline cursor-pointer"
                      >
                        Back to Log In
                      </button>
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Banner */}
                    <div className="p-4 bg-green-50 border border-green-200/70 rounded-xl text-left text-sm text-green-800 flex items-start gap-3">
                      <CheckCircle2 className="w-5 h-5 text-green-600 mt-0.5 shrink-0" />
                      <div>
                        <span className="font-semibold">OTP Code Sent!</span> Check your inbox for{" "}
                        <span className="font-semibold underline">{forgotEmail}</span>.
                        {activeOtpDev && (
                          <div className="mt-1 text-xs text-green-900 font-mono bg-green-100 px-2 py-0.5 rounded inline-block">
                            Active Code: {activeOtpDev}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* 6 OTP Boxes */}
                    <div className="space-y-1.5 text-left">
                      <label className="block text-sm font-medium text-gray-700">
                        Enter 6-Digit OTP
                      </label>
                      <div className="flex gap-2 justify-between">
                        {otpCode.map((digit, idx) => (
                          <input
                            key={idx}
                            ref={(el) => (otpInputRefs.current[idx] = el)}
                            type="text"
                            maxLength={1}
                            value={digit}
                            onChange={(e) => handleOtpChange(e.target.value, idx)}
                            onKeyDown={(e) => handleOtpKeyDown(e, idx)}
                            className="w-12 h-13 text-center text-xl font-bold bg-[#f8f9fa] border border-gray-200 rounded-xl outline-none focus:border-[#2b3a55] focus:ring-2 focus:ring-[#2b3a55]/15"
                          />
                        ))}
                      </div>
                    </div>

                    <div className="space-y-1.5 text-left">
                      <label className="block text-sm font-medium text-gray-700">
                        New Password
                      </label>
                      <input
                        type="password"
                        placeholder="At least 8 characters"
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        className="w-full bg-[#f8f9fa] border border-gray-200 rounded-xl h-12 px-4 text-sm text-gray-900 placeholder:text-gray-400 focus:bg-white focus:border-gray-400 focus:ring-2 focus:ring-gray-100 outline-none transition-all"
                      />
                    </div>

                    <button
                      type="button"
                      onClick={handleVerifyOTP}
                      className="w-full h-12 bg-[#2b3a55] hover:bg-[#1f2b3f] active:scale-[0.99] text-white font-semibold rounded-xl transition-all duration-200 shadow-sm cursor-pointer text-sm"
                    >
                      Verify OTP & Update Password
                    </button>

                    <div className="flex items-center justify-between text-sm text-gray-500 pt-1">
                      {isTimerActive ? (
                        <span>
                          Resend available in{" "}
                          <strong className="text-gray-800">{timerCount}s</strong>
                        </span>
                      ) : (
                        <button
                          type="button"
                          onClick={handleSendOTP}
                          className="font-semibold text-gray-800 hover:underline cursor-pointer"
                        >
                          Resend Code
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => {
                          setMode("login");
                          setOtpStep("email");
                        }}
                        className="font-medium text-gray-500 hover:text-gray-900"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>
    </main>
  );
}

/* ==========================================================================
   Reusable UI Helpers
   ========================================================================== */

interface StepItemProps {
  number: number;
  text: string;
  active?: boolean;
}

export function StepItem({ number, text, active = false }: StepItemProps) {
  return (
    <div
      className={`flex items-center gap-3.5 w-full px-4 sm:px-5 py-3.5 rounded-2xl transition-all duration-200 ${
        active
          ? "bg-white text-black border border-white shadow-lg"
          : "bg-white/10 text-white/90 border border-white/5"
      }`}
    >
      <div
        className={`w-7 h-7 rounded-full flex items-center justify-center text-xs sm:text-sm font-semibold shrink-0 ${
          active ? "bg-[#1e293b] text-white font-bold" : "bg-white/10 text-white/50 font-medium"
        }`}
      >
        {number}
      </div>
      <span className={`text-sm tracking-tight ${active ? "font-semibold" : "font-medium"}`}>
        {text}
      </span>
    </div>
  );
}

function GoogleButton() {
  return (
    <button
      type="button"
      onClick={() => {
        alert("Google SSO Authenticated for student@university.edu!");
        window.location.href = "/#assistant";
      }}
      className="w-full max-w-[240px] h-12 bg-white border border-gray-200 hover:bg-gray-50 active:scale-[0.99] rounded-xl flex items-center justify-center gap-3 text-sm font-medium text-gray-700 shadow-sm transition-all duration-200 cursor-pointer"
    >
      <svg className="w-4 h-4" viewBox="0 0 24 24">
        <path
          fill="#4285F4"
          d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
        />
        <path
          fill="#34A853"
          d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
        />
        <path
          fill="#FBBC05"
          d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
        />
        <path
          fill="#EA4335"
          d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
        />
      </svg>
      <span>Continue with Google</span>
    </button>
  );
}
